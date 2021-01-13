# bayesian_dnns
Practice &amp; experiment of bayesian deep neural networks with [pixyz](https://github.com/masa-su/pixyz).
## GMVAE  
A [pixyz](https://github.com/masa-su/pixyz) implementation of Gaussian Mixtured Variational Auto Encoder proposed by [Rui Sue](http://ruishu.io/2016/12/25/gmvae/).  
By using gaussian-mixtured prior for the generative model, its robustness for imbalanced data is much higher than Kingma's m2 model.  
I also refered to the [pytorch implementation by jariasf](https://github.com/jariasf/GMVAE/tree/master/pytorch)

### Example: Imbalanced MNIST
#### Data  
- **labelled**  
[label:number of images] [0:1000, 1:10, 2:10, 3:10, 4:10, 5:100, 6:70, 7:40, 8:50, 9:30]  
- **unlabelled**   
Total 50000 images. Sampling ratio of each labels are same as labelled data (imbalanced).  
- **validation**  
Total 10000 images. Sampling ratio of each labels are equal (balanced)  

#### Result
#### Kingma's M2 model
![m2_imbalanced_metrics](results/gmvae_imbalanced_mnist/m2_metrics.png)  
Latent variables(dimension 0 and 1) and reconstructed images.    
<img src="results/gmvae_imbalanced_mnist/m2_latent.png" width="300" height="300">
<img src="results/gmvae_imbalanced_mnist/m2_recon.png" width="200" height="200">  

#### GMVAE
![gmvae_imbalanced_metrics](results/gmvae_imbalanced_mnist/gmvae_metrics.png)  
Latent variables(dimension 0 and 1) and reconstructed images. You can see that each label seems to have its own distribution.  
<img src="results/gmvae_imbalanced_mnist/gmvae_latent.png" width="300" height="300">
<img src="results/gmvae_imbalanced_mnist/gmvae_recon.png" width="200" height="200"> 
