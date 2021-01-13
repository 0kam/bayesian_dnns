# bayesian_dnns
Practice &amp; experiment of bayesian deep neural networks, mainly using pixyz
## GMVAE  
Gaussian Mixed Variational Auto Encoder  
Proposed by [Rui Sue](http://ruishu.io/2016/12/25/gmvae/)
By using gaussian-mixtured prior, its robustness for imbalanced data is much better than Kingma's m2 model.
### Example: Imbalanced MNIST
- data
**labelled[label:number of train images]** 0:1000, 1:10, 2:10, 3:10, 4:10, 5:100, 6:70, 7:40, 8:50, 9:30
**unlabelled Total 50000 img. Ratio of each labels are same as labelled data.
**validation** Total 10000 img. Ratio of each labels are equal (balanced)

- result
#### Kingma's M2 model
![m2_imbalanced_metrics](results/gmvae_imbalanced_mnist/m2_metrics.png)  
![m2_imbalanced_latent](results/gmvae_imbalanced_mnist/m2_latent.png)  
![m2_imbalanced_recon](results/gmvae_imbalanced_mnist/m2_recon.png)  

#### GMVAE
![gmvae_imbalanced_metrics](results/gmvae_imbalanced_mnist/gmvae_metrics.png)
![gmvae_imbalanced_latent](results/gmvae_imbalanced_mnist/gmvae_latent.png)  
Latent variables of each labels are distributed on its own normal distribution   
![gmvae_imbalanced_recon](results/gmvae_imbalanced_mnist/gmvae_recon.png)  


