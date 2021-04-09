# NF-GAN
### Use NF-GAN to generate network flows
Configure the settings:  
The parameters for NF-GAN are stored in the file /conf/params.json.
To run the code:  
```
python seq_gan.py
```
After training, the parameters of the generator and discriminator are stored under the sub-directory /conf. The record of generated network flows is stored under the sub-directory /target. The analysis of the results is stored under the sub-directory /stats
### Distributions based on UNSW-NB15 dataset
The red line is the distribution of the generated flows whereas the blue line is the distribution of the real flows.
![distributions](./stats/kde_density.png)
