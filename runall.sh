batches=`seq 0 15`

for b in $batches
do
    echo test_batches/quantized/batch_$b
    python src/quantize.py --model_file graphs/dcgan-100.pb --inDir test_batches/quantized/batch_$b --nIter 1000 --blend --outDir test_out
done

for b in $batches
do
    echo test_batches/gaussian_noise/batch_$b
    python src/denoising.py --model_file graphs/dcgan-100.pb --inDir test_batches/gaussian_noise/batch_$b --nIter 1000 --blend --outDir test_out
done

for b in $batches
do
    echo test_batches/testset/batch_$b
    python src/colorize.py --model_file graphs/dcgan-100.pb --inDir test_batches/testset/batch_$b --nIter 1000 --blend --outDir test_out
done

for b in $batches
do
    echo test_batches/sr_linear/batch_$b
    python src/superres.py --model_file graphs/dcgan-100.pb --inDir test_batches/sr_linear/batch_$b --nIter 1000 --blend --outDir test_out
done

for b in $batches
do
    echo test_batches/sr_nn/batch_$b
    python src/superres.py --model_file graphs/dcgan-100.pb --inDir test_batches/sr_nn/batch_$b --nIter 1000 --blend --outDir test_out/sr_nn
done


