extern crate tch;
extern crate tokenizers;
use crate::tch::{IndexOp, Tensor, Kind, TchError, CModule};
use std::option::Option;
use std::io::{BufReader, BufRead, BufWriter, Write};
use std::fs::File;
use std::env;
use std::convert::{From, Into, TryInto};
extern crate fstream;

use tokenizers::{pre_tokenizers, processors};
use tokenizers::models::wordpiece::{WordPiece, WordPieceBuilder};
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::decoders;
use tokenizers::tokenizer::{Result, Tokenizer, EncodeInput, Encoding, AddedToken};
use tokenizers::utils::padding::{pad_encodings, PaddingDirection, PaddingParams, PaddingStrategy};
use tokenizers::utils::truncation::{truncate_encodings, TruncationParams, TruncationStrategy};


fn main() -> Result<()> {
    println!("Loading..");

    let args: Vec<String> = env::args().collect();
    let config: Config = parse_config(&args);
    let outfile = &config.outfile.clone();
    let batch_sz = &config.batch_size.clone();
    let batch_usz = &config.batch_usz.clone();
    let ae_model = CModule::load(&config.ae_model_path)?;
    let beam_sz: Tensor = Tensor::from(config.mutate_beam.clone());
    let tokenizer = init_tokenizer(&config)?;
    let preprocessor = Preprocessor{tokenizer, config};

    println!("Preprocessing..");
    let model_inputs: Tensor = preprocessor.preprocess();
    let mut store: Vec<Tensor> = vec![];
    let mut store_sz: i64 = 0;
    let mut outp: Tensor;

    println!("Evaluating..");
    let l = model_inputs.size()[0];
    let mut next: i64;
    let mut batch: Tensor;

    for i in (0..l).step_by(*batch_usz) {
        next = i + batch_sz;
        if next > l {
            next = l;
        }
        batch = model_inputs.i(i..next);
        outp = ae_model.forward_ts(&[&batch, &beam_sz])?;
        store_sz += &outp.size()[0];
        store.push(outp);
        println!("store_sz: {}", store_sz);
    }
    let res: Tensor = Tensor::cat(&store, 0).squeeze();
    println!("mutate size: {:?}", res.size());
    println!("Decoding..");
    let decoded = preprocessor.decode(res);

    // write to file
    println!("Writing..");
    match fstream::write_lines(outfile, decoded, true) {
        Some(b) => println!("Number of bytes written to the file: {}", b),
        None => println!("Couldn't open or write to the file!"),
      }

    Ok(())
}

enum Input {
    Single(String),
    File(String),
}

struct Config {
    input: Input,
    max_len: usize,
    vocab_file: String,
    merges_file: String,
    ae_model_path: String,
    outfile: String,
    mutate_beam: i64,
    batch_size: i64,
    batch_usz: usize,
    clear_every: i64,
}

fn parse_config(args: &[String]) -> Config {
    let input = Input::Single(String::from(&args[1]));
    // let input = Input::File(String::from(&args[1]));
    let mut mutate_beam = args[2].parse::<i64>().unwrap();
    let outfile = String::from(&args[3]);
    let max_len = 15;
    let vocab_file = String::from("xlm-bpe-vocab.json");
    let merges_file = String::from("xlm-bpe-merges.txt");
    let ae_model_path = String::from("models/sm_mutate.pt");
    let batch_size = 200;
    let batch_usz = batch_size.clone() as usize;
    let clear_every = 10000000;
    let vocab_sz = 253000;

    if mutate_beam > vocab_sz {
        mutate_beam = vocab_sz - 1;
    }
    
    Config { input, max_len, vocab_file , merges_file, ae_model_path, outfile, mutate_beam, batch_size, batch_usz, clear_every}
}

fn init_tokenizer(config: &Config) -> Result<Tokenizer> {
    let vocab_file = config.vocab_file.clone();
    let merges_file = config.merges_file.clone();
    let max_len = config.max_len.clone();

    let bpe_builder = BPE::from_files(&vocab_file, &merges_file);
    let bpe = bpe_builder.build()?;

    let mut tokenizer = Tokenizer::new(Box::new(bpe));

    tokenizer.with_post_processor(Box::new(BertProcessing::new(("</s>".to_string(),3),("<s>".to_string(), 2))));
    tokenizer.with_decoder(Box::new(decoders::bpe::BPEDecoder::default()));

    let special_tokens: Vec<&str> = vec!["<s>", "</s>", "<pad>", "<unk>"];
    let mut special_token_objects: Vec<AddedToken> = vec![];
    for token in special_tokens.iter() {
        special_token_objects.push(AddedToken{content: String::from(token.clone()), single_word: true, lstrip: true, rstrip: true});
    }
    tokenizer.add_special_tokens(&special_token_objects);

    let padding_opts = PaddingParams {
        strategy: PaddingStrategy::Fixed(max_len),
        direction: PaddingDirection::Right, 
        pad_id: 0, pad_type_id: 0, 
        pad_token: String::from("<pad>")};

    let truncation_opts = TruncationParams {
        max_length: max_len, 
        strategy: TruncationStrategy::LongestFirst, 
        stride: 1};

    tokenizer.with_padding(Option::Some(padding_opts));
    tokenizer.with_truncation(Option::Some(truncation_opts));
    Ok(tokenizer)
}

struct Preprocessor {
    tokenizer: Tokenizer,
    config: Config
}

impl Preprocessor {
    fn preprocess(&self) -> Tensor {
        let input = &self.config.input;

        let tokenized_input = match input {
            Input::Single(text) =>  self.tokenize_single(text.to_string()),
            Input::File(filename) => self.tokenize_batch(filename.to_string()),
        };

        tokenized_input.unwrap()
    }

    fn tokenize_single(&self, text: String) -> Result<Tensor> {
        let encoded_input = self.tokenizer.encode(EncodeInput::Single(text), true)?;
        let mut output = enc_to_tensor(encoded_input);
        output = output.unsqueeze(0);
        Ok(output)
    }

    fn tokenize_batch(&self, filename: String) -> Result<Tensor> {
        let mut encoded: Vec<Encoding>;
        let mut contents: Vec<EncodeInput> = self.read_inp_file(filename)?;

        encoded = self.tokenizer.encode_batch(contents, true)?;

        let ids = encoded.into_iter().map(enc_to_tensor).collect::<Vec<Tensor>>();

        let mut input: Tensor;
        input = Tensor::stack(&ids, 0);

        Ok(input)

    }

    fn read_inp_file(&self, infile: String) -> Result<Vec<EncodeInput>> {
        let mut contents: Vec<EncodeInput> = vec![];
        let file = File::open(infile)?;
        let mut reader = BufReader::new(file);
        let mut encoder_input: EncodeInput;
        for line in reader.lines() {
            encoder_input = EncodeInput::Single(line.unwrap());
            contents.push(encoder_input);
        };
        Ok(contents)
    }

    fn decode_batch(&self, input: Vec<Vec<u32>>) -> Result<Vec<String>> {
        let output = self.tokenizer.decode_batch(input, true);
        output
    }

    fn decode(&self, input: Tensor) -> Vec<String> {
        let decoder_input = self.tensor_to_vec(input);
        let output: Vec<String> = self.decode_batch(decoder_input).unwrap();
        output
    }

    fn tensor_to_vec(&self, tensor: Tensor) -> Vec<Vec<u32>> {
        let outp = Vec::<Vec::<i64>>::from(tensor);
        let outp_vec = outp.into_iter().map(i64_to_u32).collect::<Vec<Vec<u32>>>();
        outp_vec
    }

}

fn i64_to_u32(input: Vec<i64>) -> Vec<u32> {
    let u32_vec = input.into_iter().map(|x| x as u32).collect::<Vec<u32>>();
    u32_vec
}

fn enc_to_tensor(x: Encoding) -> Tensor {
    let v = x.get_ids().to_vec();
    let v_clone = v.into_iter().map(|x| x as i64).collect::<Vec<i64>>();
    let tensor = Tensor::of_slice(&v_clone);
    tensor
}