# mlvtrans

mlvtrans is a python package that implements the algorithm explained in Appendix C of the paper [Clifford gates with logical transversality for self-dual CSS codes](https://scholar.google.co.jp/citations?user=hIPtQG8AAAAJ&hl=ja). For any self-dual quantum CSS code which satisfies certain conditions (see Theorem 1. in [Clifford gates with logical transversality for self-dual CSS codes](https://scholar.google.co.jp/citations?user=hIPtQG8AAAAJ&hl=ja)), mlvtrans constructs a compatible symplectic basis, which is a symplectic basis such that (1) transversal logical Hadamard gates ![](https://latex.codecogs.com/svg.image?$\bigotimes_{j=1}^{k}\bar{H}_j$) can be implemented by transversal physical Hadamard gates ![](https://latex.codecogs.com/svg.image?$\bigotimes_{i=1}^{n}H_i$), and (2) for any ![](https://latex.codecogs.com/svg.image?$(a_1,\dots,a_k)\in\lbrace-1,1\rbrace^k$), transversal logical phase-type gates ![](https://latex.codecogs.com/svg.image?$\bigotimes_{j=1}^{k}\bar{S}_j^{a_j}$) can be implemented by transversal physical phase-type gates ![](https://latex.codecogs.com/svg.image?$\bigotimes_{i=1}^{n}S_i^{b_i}$) for some ![](https://latex.codecogs.com/svg.image?$(b_1,\dots,b_n)\in\lbrace-1,1\rbrace^n$). mlvtrans also outputs such ![](https://latex.codecogs.com/svg.image?$(b_1,\dots,b_n)\in\lbrace-1,1\rbrace^n$) for any choice of ![](https://latex.codecogs.com/svg.image?$(a_1,\dots,a_n)\in\lbrace-1,1\rbrace^n$).

# Installation

Run: `pip install mlvtrans`

# How to use mlvtrans

See [getting started notebook](https://github.com/yugotakada/mlvtrans/blob/main/getting_started.ipynb).


# Citation
```
@article{,
  doi = {},
  url = {},
  title = {Clifford gates with logical transversality for self-dual CSS codes},
  author = {},
  journal = {},
  issn = {},
  volume = ,
  pages = ,
  month = ,
  year = {2025}
}
```
