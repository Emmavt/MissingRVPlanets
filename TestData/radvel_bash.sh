#!/bin/bash
fname=$1
radvel fit -s $fname
radvel plot -t rv -s $fname
radvel mcmc -s $fname
radvel derive -s $fname
radvel plot -t derived -s $fname
shift
radvel ic -t $@ -s $fname
radvel plot -t rv corner trend -s $fname
#radvel report -s $fname
