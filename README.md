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

![image](https://github.com/user-attachments/assets/f6ed4c7e-ee1e-49ff-ac31-fb9f029c03d9)

## 計算実験

### WBICによるモデル選択
![image](https://github.com/user-attachments/assets/4e642200-38d1-4e45-a222-f67b11d84ea8)

### フィッティング結果
![image](https://github.com/user-attachments/assets/72f55425-ab75-4909-8799-19ec81765955)

### ピーク個数で周辺化したピーク位置の事後分布
![image](https://github.com/user-attachments/assets/eda36668-afb9-415f-aa40-406fd5233347)
