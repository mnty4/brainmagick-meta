#!/bin/bash

xp_string="4_4_32_meta_rec.yaml,4_4_64_meta_rec.yaml,4_4_64_nmeta_rec.yaml,4_4_128_meta_rec.yaml,4_4_128_meta_sess.yaml,4_4_128_meta_sub.yaml,4_4_128_nmeta_rec.yaml,4_4_128_nmeta_sess.yaml,4_4_128_nmeta_sub.yaml"

IFS=',' read -ra xp_array <<< "$xp_string"
for xp in "${xp_array[@]}"; do
    echo "Processing $xp"
    python -m bm.meta_learner meta_train=$xp &> "${xp%.yaml}.log"
    echo "Finished Processing $xp"
done

# nohup python -m bm.meta_learner "--multirun meta_train=4_4_32_meta_rec.yaml,4_4_64_meta_rec.yaml,4_4_64_nmeta_rec.yaml,4_4_128_meta_rec.yaml,4_4_128_meta_sess.yaml,4_4_128_meta_sub.yaml,4_4_128_nmeta_rec.yaml,4_4_128_nmeta_sess.yaml,4_4_128_nmeta_sub.yaml" &> xp.log



# nohup bash -c "python -m bm.meta_preprocess meta_preprocess=with_psd dset.n_recordings=16 \
# dset.save_dir=preprocessed_wpsd\
# &&python -m bm.meta_learner meta_train=4_4_64_meta_rec.yaml +meta_train.preprocessed_dir=preprocessed_wpsd" &> combined.log &
