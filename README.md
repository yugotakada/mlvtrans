# mlvtrans

mlvtrans is a python package that implements the algorithm explained in Appendix C of the paper [Logical Clifford gates with multilevel transversality](https://scholar.google.co.jp/citations?user=hIPtQG8AAAAJ&hl=ja). For any self-dual quantum CSS code which satisfies certain conditions (see Theorem 1. in [Logical Clifford gates with multilevel transversality](https://scholar.google.co.jp/citations?user=hIPtQG8AAAAJ&hl=ja)), mlvtrans constructs a compatible symplectic basis, which is a symplectic basis such that 

(1) transversal logical Hadamard gates $\bigotimes_{j=1}^{k} \bar{H}_j$ can be implemented by transversal physical Hadamard gates $\bigotimes_{i=1}^{n} H_i$
, 

and 

(2) for any $(a_1,\dots,a_k)\in\{-1,1\}^k$, transversal logical phase-type gates $\bigotimes_{j=1}^{k} \bar{S}_j^{a_j}$ can be implemented by transversal physical phase-type gates $\bigotimes_{i=1}^{n} S_i^{b_i}$ for some $(b_1,\dots,b_n)\in\{-1,1\}^n$.

 mlvtrans also outputs such $(b_1,\dots,b_n)\in\{-1,1\}^n$ for any choice of $(a_1,\dots,a_k)\in\{-1,1\}^k$.

# Installation

Run: `pip install mlvtrans`

# How to use mlvtrans

See [getting started notebook](https://github.com/yugotakada/mlvtrans/blob/main/getting_started.ipynb).


# Citation
```
@article{,
  doi = {},
  url = {},
  title = {Logical Clifford gates with multilevel transversality},
  author = {},
  journal = {},
  issn = {},
  volume = ,
  pages = ,
  month = ,
  year = {2025}
}
```
```
@misc{mlvtrans,
    author = {Takada, Yugo},
    license = {MIT},
    month = ,
    title = {{mlvtrans}},
    howpublished = {\url{https://github.com/yugotakada/mlvtrans}},
    version = {0.1.0},
    year = {2025}
    }
```
