# WBICによるピーク個数の推定

## ~~Widely applicable~~ Watanabe Bayesian Information Criterion; WBIC

$$
\begin{aligned}
  F_{WBIC} &= \frac{
                \int{ \mathcal{L}(\Theta)P(\mathcal{D}|\Theta)^{\beta}P(\Theta)\mathrm{d}\Theta }
              }{
                \int{ P(\mathcal{D}|\Theta)^{\beta}P(\Theta)\mathrm{d}\Theta }
              }\\
  &= \mathbb{E}[\mathcal{L}(\Theta)]_{P(\mathcal{D}|\Theta)^{\beta}P(\Theta)},
  \quad \text{where} \quad \beta^{-1} = \ln{N}
\end{aligned}
$$

## トイデータ

![data](https://github.com/user-attachments/assets/a00186c2-6fcf-485d-a5f5-f22ac95c0036)

## 計算実験

### WBICによるモデル選択

![wbic](https://github.com/user-attachments/assets/eb90503b-a191-44aa-8391-85aaa0804259)


### フィッティング結果

![fitting_L_003](https://github.com/user-attachments/assets/e6e0b702-80e8-414a-8872-2b17487993cb)![fitting_L_004](https://github.com/user-attachments/assets/5dea02cd-7bb1-4a72-8632-25e8a339ca3e)

