use ndarray::{Array, Array1, Array2, Array3, Axis, s};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use std::io;
use std::io::Write;

type Parameters<'a> = (&'a Array2<f64>, &'a Array2<f64>, &'a Array1<f64>);


pub fn baum_welch(
    observes: &Vec<usize>, 
    obs_states: usize, 
    hidden_states: usize, 
    iterations: usize,
    mutable_a: bool,
    s_matrix: Option<&Array2<f64>>
) -> (Array2<f64>, Array2<f64>, Array1<f64>) {
    let mut a = match s_matrix {
        Some(val) => val.to_owned(),  
        None => init_matrix(hidden_states, hidden_states)
    };
    let mut b = init_matrix(hidden_states, obs_states);
    let mut m = init_matrix(1, hidden_states).into_shape(hidden_states).expect("Incorr shape");
    // let mut old_log_prob = -f64::INFINITY; 
    for i in 0..iterations {
        print!("Current iteration: {}/{}\r", i, iterations);
        io::stdout().flush().expect("error");
        let (alpha, c) = alpha_pass((&a, &b, &m), &observes, hidden_states);
        // println!("{}", alpha);
        let beta = beta_pass((&a, &b, &m), &c, &observes, hidden_states);
        let (gamma, digamma) = gamma_digamma(&alpha, &beta, (&a, &b, &m), &observes, hidden_states);
        if mutable_a {
            (a, b, m) = reestimate(&gamma, &digamma, &observes, hidden_states, obs_states);
        } else {
            (_, b, m) = reestimate(&gamma, &digamma, &observes, hidden_states, obs_states);
        }
        // let mut log_prob = 0.0;
        // for t in 0..observes.len() {
        //     log_prob += f64::ln(1.0 / c[t]); 
        // }
        // log_prob = -log_prob;
        // let delta = f64::abs(log_prob - old_log_prob);
        if i > iterations {
            break;
        }
        // old_log_prob = log_prob;
    }
    println!("\n");
    (a, b, m)
}

pub fn viterbi(p: Parameters, observes: &Vec<usize>, states: usize) -> Vec<usize> {
    let (a, b, m) = p;
    let obs_len = observes.len();
    let mut delta = Array2::<f64>::from_elem((obs_len, states), f64::MIN);
    let mut arg_delta = Array2::<usize>::zeros((obs_len, states));
    
    for i in 0..states {
        if m[i] != 0.0 {
            delta[[0, i]] = f64::ln(m[i]) + f64::ln(b[[i, observes[0]]]);
            arg_delta[[0, i]] = 0;
        }
    }

    for t in 1..obs_len {
        for i in 0..states {
            for j in 0..states {
                let val = delta[[t-1, j]] + f64::ln(a[[i, j]]) + f64::ln(b[[i, observes[t]]]);
                if val > delta[[t, i]] { 
                    delta[[t, i]] = val;
                    arg_delta[[t, i]] = j;
                }
            }
        }
    }

    let mut z = vec![0; obs_len];
    z[obs_len - 1] = delta.row(obs_len-1).iter()
            .enumerate()
            .reduce(|(i1, a), (i2, b)| if a > b { (i1, a) } else { (i2, b) })
            .unwrap().0;
    for t in (1..obs_len).rev() {
        z[t-1] = arg_delta[[t, z[t]]];
    }
    
    return z
}

fn init_matrix(size_x: usize, size_y: usize) -> Array2<f64> {
    let fill_value = 1. / (size_y as f64);
    let distr = Normal::new(fill_value, 0.001)
        .expect("Error encountered while creating Normal distr!!");
    
    let mtrx = Array::random((size_x, size_y), distr);
    let sum = &mtrx.sum_axis(Axis(1));
    (&mtrx.t() / sum).t().to_owned()
}

fn alpha_pass(p: Parameters, observes: &Vec<usize>, states: usize) -> (Array2<f64>, Array1<f64>) {
    let (a, b, m) = p;
    let obs_len = observes.len();
    let mut c = Array1::<f64>::zeros(obs_len);
    let mut alpha = Array2::<f64>::zeros((obs_len, states));
    
    for i in 0..states {
        alpha[[0, i]] = m[i] * b[[i, observes[0]]];
        c[0] += alpha[[0, i]];
    }
    for i in 0..states {
        alpha[[0, i]] /= c[0]
    }

    for t in 1..obs_len {
        for i in 0..states {
            for j in 0..states {
                alpha[[t, i]] += alpha[[t-1, j]]*a[[j, i]];
            }
            alpha[[t, i]] *= b[[i, observes[t]]];
            c[t] += alpha[[t, i]];
        }
        for i in 0..states {
            alpha[[t, i]] /= c[t];
        }
    }

    (alpha, c)
}

fn beta_pass(p: Parameters, c: &Array1<f64>,  observes: &Vec<usize>, states: usize) -> Array2<f64> {
    let (a, b, _) = p;
    let obs_len = observes.len();
    let mut beta = Array2::<f64>::zeros((obs_len, states));
    
    for i in 0..states {
        beta[[obs_len-1, i]] = c[obs_len - 1];
    }

    for t in (0..obs_len-1).rev() {
        for i in 0..states {
            beta[[t, i]] = 0.0;
            for j in 0..states {
                beta[[t, i]] +=  a[[i, j]] * b[[j, observes[t+1]]] * beta[[t+1, j]]; 
            }
            beta[[t, i]] /= c[t];
        }
    }

    beta
}

fn gamma_digamma(
    alpha: &Array2<f64>, 
    beta: &Array2<f64>, 
    p: Parameters, 
    observes: &Vec<usize>, 
    states: usize
) -> (Array2<f64>, Array3<f64>) {
    let (a, b, _) = p;
    let obs_len = observes.len();
    let mut digamma = Array3::<f64>::zeros((obs_len, states, states));
    let mut gamma = Array2::<f64>::zeros((obs_len, states));

    for t in 0..obs_len-1 {
        let mut denom = 0.0;
        for i in 0..states {
            for j in 0..states {
                denom += alpha[[t, i]] * a[[i, j]] * b[[j, observes[t+1]]] * beta[[t+1, j]];
            }
        }
        for i in 0..states {
            gamma[[t, i]] = 0.0;
            for j in 0..states {
                digamma[[t, i, j]] = (alpha[[t, i]] * a[[i, j]] * b[[j, observes[t+1]]] * beta[[t+1, j]]) / denom;
                gamma[[t, i]] += digamma[[t, i, j]];
            }
        }
    }
    (gamma, digamma)
}

fn reestimate(
    gamma: &Array2<f64>,
    digamma: &Array3<f64>,
    observes: &Vec<usize>,
    states: usize,
    obs_states: usize
) -> (Array2<f64>, Array2<f64>, Array1<f64>) {
    let obs_len = observes.len();
    let m = gamma.row(0).to_owned();
    let mut a = Array2::<f64>::zeros((states, states));
    a.assign(&(
        &digamma.slice(s![..obs_len-1, .., ..]).sum_axis(Axis(0)).t() 
        /
        &gamma.slice(s![..obs_len-1, ..]).sum_axis(Axis(0)) 
    ).t());
    let mut b = Array2::<f64>::zeros((states, obs_states));
    for i in 0..states {
        for j in 0..obs_states {
            let denom = gamma.column(i).sum();
            b[[i, j]] = gamma.column(i).iter().enumerate()
                .fold(0.0, |acc, (t, val)| if observes[t] == j { acc + val } else { acc })
                / denom;
        }
    }

    (a, b, m)
}


// #[cfg(test)]
// mod tests {
//     use super::*;
//     
//     #[test]
//     fn init() {
//         init_matrix(3, 3);
//     }
//
//     #[test]
//     fn alpha() {
//         let mut mtrx1 = init_matrix(3, 3);
//         let mut mtrx2 = init_matrix(3, 3);
//         let mut m = init_matrix(1, 3).into_shape(3).expect("Incorr shape");
//         let (alpha, c) = alpha_pass((&mtrx1, &mtrx2, &m), &vec![1, 2, 0, 2, 1], 3);
//         let beta = beta_pass((&mtrx1, &mtrx2, &m), &c, &vec![1, 2, 0, 2, 1], 3);
//         let (gamma, digamma) = gamma_digamma(&alpha, &beta, (&mtrx1, &mtrx2, &m), &vec![1, 2, 0, 2, 1], 3);
//         (mtrx1, mtrx2, m) = reestimate(&gamma, &digamma, &vec![1, 2, 0, 2, 1], 3, 3);
//         println!("{}", mtrx1);
//     }
// }
