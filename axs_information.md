# Some Notes about working with AXS

From our session on 2019-08-27.

* We're working off a notebook that Eric will add to the repo [ADD LINK HERE!]().
* Eric has an ssh tunnel command that allows us to look at the Spark interface to know what
spark is doing. 
* copy Colin's code for starting up Spark, if needed, fiddle with the numbers
* make sure to use your own directory for storing files, because Spark tends to overwrite stuff!
* there isn't an up-to-date table of all the catalogue that are in AXS, but as of 2019-08-26, the `ztf_mar18_all`
table should be up to date with ZTF data
* Detailed information about the schema can be found on [this website](https://github.com/dirac-institute/alert_tools/tree/master/ZTF10).
* if you use `toPandas`, it pulls everything into memory, and that can make things crash horribly!
* Need to use `.collect()` for spark queries to tell it to actually execute the query
* can do filtering + selections on parameters, even if they're not returned
* can also cross-match with other catalogues, but those catalogues need to be in AXS (which is not super hard)
  
