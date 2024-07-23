# WBICによるピーク個数の推定

## ~~Widely applicable~~ Watanabe Bayesian Information Criterion; WBIC
観測データ$`\mathcal{D}`$が与えられたとき，WBICは下記の式で与えられる：

$$
\begin{aligned}
  F_{WBIC} &= \frac{
                \int{ n \mathcal{L}(\Theta)P(\mathcal{D}|\Theta)^{\beta}P(\Theta)\mathrm{d}\Theta }
              }{
                \int{ P(\mathcal{D}|\Theta)^{\beta}P(\Theta)\mathrm{d}\Theta }
              }\\
  &= \mathbb{E}[n \mathcal{L}(\Theta)]_{P(\mathcal{D}|\Theta)^{\beta}P(\Theta)},
  \quad \text{where} \quad \beta^{-1} = \log{n}
\end{aligned}
$$

## ベイズ自由エネルギー

$$
\begin{aligned}
  F_n &= - \log{Z_n}\\
    &= - \log{ \int{ \exp{\\{-n\mathcal{L}(\Theta)\\}p(\Theta)} \mathrm{d} \Theta } }\\
    &= - n \mathcal{L}(\Theta_{*}) + \lambda \log{n} - (m-1) \log{\log{n}} + \mathcal{O}_{P}(n^{-1}).
\end{aligned}
$$

ここで，逆温度$`\beta`$を導入する：

$$
\begin{aligned}
  F_{n\beta} &= - \log{Z_{n\beta}}\\
    &= - \log{ \int{ \exp{\\{- n \beta \mathcal{L}(\Theta)\\}p(\Theta)} \mathrm{d} \Theta } }\\
    &= - n\beta \mathcal{L}(\Theta_{*}) + \lambda \log{n\beta} - (m-1) \log{\log{n\beta}} + \mathcal{O}_{P}((n\beta)^{-1}).
\end{aligned}
$$

逆温度$`\beta`$の勾配を考えると：

$$
\begin{aligned}
  \frac{\partial F_{n\beta}}{\partial \beta} = - n \mathcal{L}(\Theta_{*}) + \frac{\lambda}{\beta} + C
\end{aligned}
$$

である．$`\beta^{-1} = \log{n}`$とおくと，

$$
  \left. \frac{\partial F_{n\beta}}{\partial \beta} \right|\_{\beta^{-1} = \log{n}} = - n \mathcal{L}(\Theta_{*}) + \lambda \log{n} + C
$$

つまり，

$$
  F_n \sim \left. \frac{\partial F_{n\beta}}{\partial \beta} \right|\_{\beta^{-1} = \log{n}}
$$

であり，WBICはこの値を計算している．

$`\frac{\partial F_{n\beta}}{\partial \beta}`$は$`F_{n\beta}`$の対数を取ったあとの$`\beta`$の微分操作から：

$$
  \frac{\partial F_{n\beta}}{\partial \beta} = \mathbb{E}[n \mathcal{L}(\Theta)]_{P(\mathcal{D}|\Theta)^{\beta}P(\Theta)},
$$

で計算できる．

## トイデータ

![data](https://github.com/user-attachments/assets/a00186c2-6fcf-485d-a5f5-f22ac95c0036)

## 計算実験

### WBICによるモデル選択

![wbic](https://github.com/user-attachments/assets/eb90503b-a191-44aa-8391-85aaa0804259)


### フィッティング結果

![fitting_L_003](https://github.com/user-attachments/assets/e6e0b702-80e8-414a-8872-2b17487993cb)![fitting_L_004](https://github.com/user-attachments/assets/5dea02cd-7bb1-4a72-8632-25e8a339ca3e)

