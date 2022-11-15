


// Number of times to run inference and calculate average time
// use cuda event to record key points

 constexpr int ITERATIONS = 10;
 bool SimpleOnnx::infer()
 {
     CudaEvent start;
     CudaEvent end;
     double totalTime = 0.0;
     CudaStream stream;
     for (int i = 0; i < ITERATIONS; ++i)
     {
         float elapsedTime;
         // Measure time it takes to copy input to GPU, run inference and move output back to CPU.
         cudaEventRecord(start, stream);
         launchInference(mContext.get(), stream, mInputTensor, mOutputTensor, mBindings, mParams.batchSize);
         cudaEventRecord(end, stream);
         // Wait until the work is finished.
         cudaStreamSynchronize(stream);
         cudaEventElapsedTime(&elapsedTime, start, end);
         totalTime += elapsedTime;
     }
     cout << "Inference batch size " << mParams.batchSize << " average over " << ITERATIONS << " runs is " << totalTime / ITERATIONS << "ms" << endl;
     return true;
 }