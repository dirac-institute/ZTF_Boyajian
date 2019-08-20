# ZTF_Boyajian

If the "Boyajian's Star" phenomenon is a common feature in F stars, then we should see *many* of them in the current ZTF survey. James Davenport & John Ruan [speculated about this](https://beta.briefideas.org/ideas/534f2373fdf0cd3de184f11a63c4a3ee) in 2016 with regards to SDSS Stripe 82, and [recent efforts](https://ui.adsabs.harvard.edu/abs/2019ApJ...880L...7S/) to find "dippers" in ASAS seem encouraging.

So let's use [AXS](https://github.com/dirac-institute/AXS) to explore the ZTF data, following [tutorials](https://github.com/ctslater/ztf_experiments/blob/master/axs_ztf_demo.ipynb), and with the help of our colleagues at DIRAC.

1. Use the Gaia DR2 + ZTF photometry to pick all the F dwarfs in ZTF of interest 
2. For each light curve, pull some high-level stats (e.g. Range, 5th-95th percentile, etc)
3. For any high-variability candidates, save their light curves for inspection


This project will result in a highly useful notebook, ideal for demonstrating ZTF interrogation for many projects. We also should have a short but useful publication as a result, that could place a new upper-limit on the occurance rate of Boyajian's Star-like activity from F dwarfs.
