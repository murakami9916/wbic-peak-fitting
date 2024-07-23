# WBICによるピーク個数の推定

## ~~Widely applicable~~ Watanabe Bayesian Information Criterion; WBIC
観測データ$`\mathcal{D}=\{(x_i, y_i)\}_{i=1}^{n}`$が与えられたとき，WBICは下記の式で与えられる：

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

ここで，$`\mathcal{L}(\Theta)`$はパラメータ$`\Theta`$における負の対数尤度である：

$$
  \mathcal{L}(\Theta) = - \frac{1}{n} \sum_{i=1}^{n}{ \log{ P(y_{i}|\Theta) } }
$$

## ベイズ自由エネルギーとWBIC
データ点数が$`n`$のデータ$`\mathcal{D}`$について，ベイズ自由エネルギー$`F_n`$を考える．

$$
\begin{aligned}
  F_n &= - \log{Z_n}\\
    &= - \log{ \int{ \exp{\\{-n\mathcal{L}(\Theta)\\}P(\Theta)} \mathrm{d} \Theta } }\\
    &= - n \mathcal{L}(\Theta_{*}) + \lambda \log{n} - (m-1) \log{\log{n}} + \mathcal{O}_{P}(n^{-1}).
\end{aligned}
$$

ここで，$`\lambda`$は(代数幾何の)実対数閾値であり，渡辺ベイズ理論では学習係数とも呼ばれる．もし正則なモデルであればパラメータの次元数$`d`$は$`d=2\lambda`$である．

ここで，逆温度$`\beta`$を導入する：

$$
\begin{aligned}
  F_{n}(\beta) &= - \log{Z_{n}(\beta)}\\
    &= - \log{ \int{ \exp{\\{- n \beta \mathcal{L}(\Theta)\\}P(\Theta)} \mathrm{d} \Theta } }\\
    &= - n\beta \mathcal{L}(\Theta_{*}) + \lambda \log{n\beta} - (m-1) \log{\log{n\beta}} + \mathcal{O}_{P}((n\beta)^{-1}).
\end{aligned}
$$

逆温度$`\beta`$の偏微分$`F'_{n}(\beta)`$を考えると：

$$
\begin{aligned}
  \frac{\partial F_{n}(\beta)}{\partial \beta} = - n \mathcal{L}(\Theta_{*}) + \frac{\lambda}{\beta} + \mathcal{O}(\sqrt{\log{n}}),
\end{aligned}
$$

である．$`\beta^{-1} = \log{n}`$とおくと，

$$
  \left. \frac{\partial F_{n\beta}}{\partial \beta} \right|\_{\beta^{-1} = \log{n}} = - n \mathcal{L}(\Theta_{*}) + \lambda \log{n} + \mathcal{O}(\sqrt{\log{n}}),
$$

である．つまり，

$$
  F_n \approx \left. \frac{\partial F_{n\beta}}{\partial \beta} \right|\_{\beta^{-1} = \log{n}}
$$

であり，WBICはこの値を計算している．

逆温度での偏微分$`F'_{n}(\beta)`$は$`F_{n}(\beta)`$の対数を取ったあとの$`\beta`$の微分操作から：

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

