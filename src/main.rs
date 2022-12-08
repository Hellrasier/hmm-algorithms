mod hmm;
use std::fs;
use clap::Parser;
use regex::Regex;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    method: String,
    #[arg(long)]
    length: Option<usize>,
    #[arg(long)]
    states: Option<usize>,
    #[arg(long)]
    learn_size: Option<usize>,
    #[arg(long)]
    iterations: Option<usize>,
}


fn main() {
    let raw = fs::read_to_string("./text.txt")
        .expect("Error, cannot open a file");
    let abc = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя";
    let re = Regex::new(r"[^а-яА-ЯіІїЇєЄ]").unwrap();
    let text = re
        .replace_all(&raw, "")
        .to_string()
        .to_lowercase()
        .replace("\n", "");
    let encoded = fs::read_to_string("./encrypted.txt")
        .expect("No such file encrypted.txt");

    let cli = Cli::parse();
    if cli.method == String::from("analysis") {
        let len = cli.length
            .expect("Error, --length argument should be provided");
        let states = cli.states
            .expect("Error, --states argument should be provided"); 
        let iterations = cli.states
            .expect("Error, --iterations argument should be provided"); 
        let observes = text.chars()
            .take(len)
            .map(
                |chr| match abc.chars().position(|x| x == chr) {
                    Some(indx) => indx as usize,
                    None => 0
                }
            )
            .collect::<Vec<usize>>();
        text_analysis(observes, abc, states, iterations);
    } else if cli.method == String::from("encode") {
        let len = cli.length
            .expect("Error, --length argument should be provided");
        let encoded = text_encode(text, abc, len);
        fs::write("./encrypted.txt", encoded).unwrap();
    } else {
        let len = cli.length
            .expect("Error, --length argument should be provided");
        let learn_size = cli.learn_size
            .expect("Error, --learn_size argument should be provided"); 
        let iterations = cli.iterations
            .expect("Error, --iterations argument should be provided"); 
        text_decode(text, encoded, abc, len, learn_size, iterations);
    }

}

fn text_analysis(observes: Vec<usize>, abc: &str, states: usize, iterations: usize) {
    let (a, b, m) = hmm::baum_welch(&observes, abc.chars().count(), states, iterations, true, None);
    println!("Matrix A: {:.5}", a);
    println!("Matrix B: {:.5}", b);
    println!("Vector: {:.5}", m);
    println!("Clusterizing into states:");
    let mut clusters: Vec<Vec<char>> = vec![Vec::new(); states];
    for (i, probs) in b.axis_iter(ndarray::Axis(1)).enumerate() {
        let (state, _) = probs.iter()
            .enumerate()
            .reduce(|(i1, a), (i2, b)| if a > b { (i1, a) } else { (i2, b) })
            .unwrap();
        clusters[state].push(abc.chars().nth(i).unwrap());
    }
    for (i, cluster) in clusters.iter().enumerate() {
        println!("State {} letters - {:?}", i + 1, cluster);
    }
}

fn text_encode(text: String, abc: &str, len: usize) -> String {
    let shift = 5;
    text.chars()
        .take(len)
        .map(
            |chr| {
                let pos = abc.chars().position(|x| x == chr).unwrap();
                abc.chars().nth((pos + shift) % abc.chars().count()).unwrap()
            }
        )
        .collect::<String>()
}

fn text_decode(text: String, encoded: String, abc: &str, len: usize, learn_size: usize, iterations: usize) {
    let symbols = text.chars()
        .take(200000)
        .map(
            |chr| match abc.chars().position(|x| x == chr) {
                Some(indx) => indx as usize,
                None => 0
            }
        )
        .collect::<Vec<usize>>();
    let encoded_symbols = encoded.chars()
        .take(len)
        .map(
            |chr| match abc.chars().position(|x| x == chr) {
                Some(indx) => indx as usize,
                None => 0
            }
        )
        .collect::<Vec<usize>>();
    let abc_len = abc.chars().count();
    let mut a = ndarray::Array2::<f64>::from_elem((abc_len, abc_len), 5.0);
    
    symbols.iter()
        .reduce(|i, j| {
            a[[*i, *j]] += 1.0;
            j
        });
    for i in 0..abc_len {
        let sum_row = a.row(i).sum();
        for j in 0..abc_len {
            a[[i, j]] /= sum_row;
        }
    }

    let (_, b, m) = hmm::baum_welch(&encoded_symbols[..2000].to_vec(), abc_len, abc_len, iterations, false, Some(&a));
    let z = hmm::viterbi((&a, &b, &m), &encoded_symbols, abc_len);
    let decrypted = z.iter().map(|&i| abc.chars().nth(i).unwrap()).collect::<String>();
    let recognized = z.iter()
        .enumerate()
        .fold(0, |acc, (i, chr)| {
            if chr == &symbols[i] {
                acc + 1
            } else {
                acc
            }
        }) as f32 / z.len() as f32;
    println!("Percent of dectypted is: {}%", recognized * 100.0);
    println!("Decrypted text in dectypted.txt");
    fs::write("decrypted.txt", decrypted).unwrap();
}
