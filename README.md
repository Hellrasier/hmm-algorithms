# HMM Baum-welch and Viterbi algorithm written on rust

To run:

```console
$ cargo run -r [method] [args]
```
**methods**: `analysis`, `encode`, `decode`

`analysis` does text clusterization using Baum-welch algorithm of text in *text.txt* file 
args: `--length` for text length and `--states` for number of clusters

`encode` encrypts text from text.txt file with Ceaser cipher and puts encoded into encrypted.txt

`decode` decrypts text from encrypted.txt using Baum-welch and Viterbi algorithm file and puts decrypted to decrypted.txt
args: `--length` for decrypting length, `--learn_size` for first n characters put in Baum-welch algorithm, `--iterations` for iterations of Baum-welch algorithm. 
