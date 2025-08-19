# Using the Serialized FlexControlTripleStreamsVAE Model in C++

## Prerequisites

```cpp
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
```

1. Loading the Model

```cpp
// Load the serialized TorchScript model
torch::jit::script::Module model;
try {
    model = torch::jit::load("path/to/your/triplestreams_vae_model.pt");
    model.eval(); // Set to evaluation mode
    std::cout << "Model loaded successfully!" << std::endl;
} catch (const c10::Error& e) {
    std::cerr << "Error loading the model: " << e.msg() << std::endl;
    return -1;
}
```

2. Input Preparation
HVO Input (Always Required)
```cpp
// Create HVO input tensor: [batch_size=1, sequence_length=32, features=3]
// Features: [hits, velocities, offsets]
torch::Tensor hvo_input = torch::rand({1, 32, 3});

// Example with real data:
// float hvo_data[32][3] = { /* your drum pattern data */ };
// torch::Tensor hvo_input = torch::from_blob(hvo_data, {1, 32, 3}, torch::kFloat);
Control Tokens Preparation
```

Case 1: No Controls Available
```cpp
// When no encoding controls are configured
torch::Tensor encoding_controls = torch::empty({1, 0}, torch::kFloat);

// When no decoding controls are configured  
torch::Tensor decoding_controls = torch::empty({1, 0}, torch::kFloat);

```

Case 2: Mixed Discrete and Continuous Controls
```cpp
// Example configuration:
// encoding_controls: [discrete(13 tokens), continuous, discrete(10 tokens)]
// decoding_controls: [continuous, discrete(8 tokens)]

// Encoding controls: [discrete_value, continuous_value, discrete_value]
std::vector<float> enc_values = {5.0f, 0.7f, 2.0f}; // discrete=5, continuous=0.7, discrete=2
torch::Tensor encoding_controls = torch::tensor(enc_values, torch::kFloat).reshape({1, 3});

// Decoding controls: [continuous_value, discrete_value]
std::vector<float> dec_values = {0.3f, 4.0f}; // continuous=0.3, discrete=4
torch::Tensor decoding_controls = torch::tensor(dec_values, torch::kFloat).reshape({1, 2});

```

Case 3: All Discrete Controls
```cpp
// All controls are discrete tokens (integers)
std::vector<float> enc_discrete = {1.0f, 5.0f, 8.0f}; // Will be cast to long internally
torch::Tensor encoding_controls = torch::tensor(enc_discrete, torch::kFloat).reshape({1, 3});

std::vector<float> dec_discrete = {2.0f, 7.0f};
torch::Tensor decoding_controls = torch::tensor(dec_discrete, torch::kFloat).reshape({1, 2});
Case 4: All Continuous Controls
```cpp// All controls are continuous values [0, 1]
std::vector<float> enc_continuous = {0.2f, 0.8f, 0.5f};
torch::Tensor encoding_controls = torch::tensor(enc_continuous, torch::kFloat).reshape({1, 3});

std::vector<float> dec_continuous = {0.6f, 0.1f, 0.9f};
torch::Tensor decoding_controls = torch::tensor(dec_continuous, torch::kFloat).reshape({1, 3});

```

3. Model Inference Methods
Method 1: Full Forward Pass
```cpp
std::vector<torch::jit::IValue> forward_inputs;
forward_inputs.push_back(hvo_input);
forward_inputs.push_back(encoding_controls);
forward_inputs.push_back(decoding_controls);

// Call forward method: returns [h_logits, v_logits, o_logits, mu, log_var, latent_z]
auto forward_output = model.get_method("forward")(forward_inputs);
auto output_tuple = forward_output.toTuple();

torch::Tensor h_logits = output_tuple->elements()[0].toTensor();  // [1, 32, 1] 
torch::Tensor v_logits = output_tuple->elements()[1].toTensor();  // [1, 32, 1]
torch::Tensor o_logits = output_tuple->elements()[2].toTensor();  // [1, 32, 1]
torch::Tensor mu = output_tuple->elements()[3].toTensor();        // [1, latent_dim]
torch::Tensor log_var = output_tuple->elements()[4].toTensor();   // [1, latent_dim]
torch::Tensor latent_z = output_tuple->elements()[5].toTensor();  // [1, latent_dim]
```

Method 2: Prediction (Recommended for Generation)
```cpp
std::vector<torch::jit::IValue> predict_inputs;
predict_inputs.push_back(hvo_input);
predict_inputs.push_back(encoding_controls);
predict_inputs.push_back(decoding_controls);
predict_inputs.push_back(0.5); // threshold (optional, default=0.5)

// Call predict method: returns [hvo_output, latent_z]
auto predict_output = model.get_method("predict")(predict_inputs);
auto predict_tuple = predict_output.toTuple();

torch::Tensor hvo_generated = predict_tuple->elements()[0].toTensor();  // [1, 32, 3]
torch::Tensor latent_z = predict_tuple->elements()[1].toTensor();       // [1, latent_dim]

// Extract individual components
torch::Tensor hits = hvo_generated.slice(2, 0, 1);      // [1, 32, 1]
torch::Tensor velocities = hvo_generated.slice(2, 1, 2); // [1, 32, 1] 
torch::Tensor offsets = hvo_generated.slice(2, 2, 3);    // [1, 32, 1]

```

Method 3: Encode Only
```cpp
std::vector<torch::jit::IValue> encode_inputs;
encode_inputs.push_back(hvo_input);
encode_inputs.push_back(encoding_controls);

// Call encodeLatent method: returns [mu, log_var, latent_z, memory]
auto encode_output = model.get_method("encodeLatent")(encode_inputs);
auto encode_tuple = encode_output.toTuple();

torch::Tensor mu = encode_tuple->elements()[0].toTensor();
torch::Tensor log_var = encode_tuple->elements()[1].toTensor(); 
torch::Tensor latent_z = encode_tuple->elements()[2].toTensor();
torch::Tensor memory = encode_tuple->elements()[3].toTensor();

```

Method 4: Decode Only
```cpp
// Use latent_z from encoding or create random latent
torch::Tensor random_latent = torch::randn({1, latent_dim}); // Get latent_dim from model config

std::vector<torch::jit::IValue> decode_inputs;
decode_inputs.push_back(random_latent);
decode_inputs.push_back(decoding_controls);

// Call decode method: returns [h_logits, v_logits, o_logits, hvo_logits]
auto decode_output = model.get_method("decode")(decode_inputs);
auto decode_tuple = decode_output.toTuple();

torch::Tensor h_logits = decode_tuple->elements()[0].toTensor();
torch::Tensor v_logits = decode_tuple->elements()[1].toTensor();
torch::Tensor o_logits = decode_tuple->elements()[2].toTensor();
torch::Tensor hvo_logits = decode_tuple->elements()[3].toTensor();
```

Method 5: Advanced Sampling

```cpp
std::vector<torch::jit::IValue> sample_inputs;
sample_inputs.push_back(latent_z);
sample_inputs.push_back(decoding_controls);

// Voice thresholds for each drum component [kick, snare, hihat]
torch::Tensor voice_thresholds = torch::tensor({0.5f, 0.6f, 0.4f});
sample_inputs.push_back(voice_thresholds);

// Maximum number of hits allowed per voice
torch::Tensor max_counts = torch::tensor({8, 6, 16}); // Max hits per 32 steps
sample_inputs.push_back(max_counts);

sample_inputs.push_back(0); // sampling_mode: 0=threshold, 1=probabilistic

// Call sample method: returns [hits, velocities, offsets]
auto sample_output = model.get_method("sample")(sample_inputs);
auto sample_tuple = sample_output.toTuple();

torch::Tensor hits = sample_tuple->elements()[0].toTensor();
torch::Tensor velocities = sample_tuple->elements()[1].toTensor();
torch::Tensor offsets = sample_tuple->elements()[2].toTensor();
```

4. Complete Example: Drum Pattern Generation

```cpp
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

int main() {
    // Load model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("triplestreams_model.pt");
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error: " << e.msg() << std::endl;
        return -1;
    }

    // Create input HVO pattern (32 steps, 3 voices)
    torch::Tensor hvo_input = torch::rand({1, 32, 3});
    
    // Example: Model with 2 encoding controls [discrete(13), continuous]
    //          and 3 decoding controls [continuous, discrete(10), continuous]
    
    // Encoding controls: genre=5 (discrete), energy=0.8 (continuous)  
    torch::Tensor encoding_controls = torch::tensor({5.0f, 0.8f}, torch::kFloat).reshape({1, 2});
    
    // Decoding controls: tempo=0.6, style=3, complexity=0.4
    torch::Tensor decoding_controls = torch::tensor({0.6f, 3.0f, 0.4f}, torch::kFloat).reshape({1, 3});

    // Generate new drum pattern
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(hvo_input);
    inputs.push_back(encoding_controls);
    inputs.push_back(decoding_controls);
    inputs.push_back(0.5); // threshold

    try {
        auto output = model.get_method("predict")(inputs);
        auto result_tuple = output.toTuple();
        
        torch::Tensor generated_hvo = result_tuple->elements()[0].toTensor();
        torch::Tensor latent_z = result_tuple->elements()[1].toTensor();
        
        std::cout << "Generated HVO shape: " << generated_hvo.sizes() << std::endl;
        std::cout << "Latent vector shape: " << latent_z.sizes() << std::endl;
        
        // Convert to CPU and access data
        auto hvo_cpu = generated_hvo.cpu();
        auto accessor = hvo_cpu.accessor<float, 3>();
        
        std::cout << "\nGenerated drum pattern (first 8 steps):" << std::endl;
        std::cout << "Step | Kick | Snare | HiHat |" << std::endl;
        std::cout << "-----+------+-------+-------|" << std::endl;
        
        for (int step = 0; step < 8; ++step) {
            printf("%4d |%5.2f |%6.2f |%6.2f |\n", 
                   step,
                   accessor[0][step][0], // hits
                   accessor[0][step][1], // velocities  
                   accessor[0][step][2]  // offsets
            );
        }
        
    } catch (const c10::Error& e) {
        std::cerr << "Inference error: " << e.msg() << std::endl;
        return -1;
    }

    return 0;
}
```

5. Handling Different Control Configurations

No Encoding Controls

```cpp
// When model has no encoding controls
torch::Tensor encoding_controls = torch::empty({1, 0}, torch::kFloat);
```

No Decoding Controls

```cpp// When model has no decoding controls
torch::Tensor decoding_controls = torch::empty({1, 0}, torch::kFloat);
```

No Controls At All
```cpp
// When model has no controls (pure VAE)
torch::Tensor encoding_controls = torch::empty({1, 0}, torch::kFloat);
torch::Tensor decoding_controls = torch::empty({1, 0}, torch::kFloat);

// Usage remains the same
std::vector<torch::jit::IValue> inputs;
inputs.push_back(hvo_input);
inputs.push_back(encoding_controls);  // Empty tensor
inputs.push_back(decoding_controls);  // Empty tensor
```

6. Error Handling Best Practices

```cpp

bool validateControlDimensions(const torch::Tensor& controls, int expected_controls) {
    if (controls.size(1) != expected_controls) {
        std::cerr << "Expected " << expected_controls << " controls, got " 
                  << controls.size(1) << std::endl;
        return false;
    }
    return true;
}

// Validate discrete control ranges
bool validateDiscreteControl(float value, int max_tokens) {
    int int_val = static_cast<int>(value);
    if (int_val < 0 || int_val >= max_tokens) {
        std::cerr << "Discrete control value " << int_val 
                  << " out of range [0, " << max_tokens << ")" << std::endl;
        return false;
    }
    return true;
}

// Validate continuous control ranges  
bool validateContinuousControl(float value) {
    if (value < 0.0f || value > 1.0f) {
        std::cerr << "Continuous control value " << value 
                  << " out of range [0.0, 1.0]" << std::endl;
        return false;
    }
    return true;
}
```

7. Control Modes and Their Effects
8. 
Control Mode Summary

**prepend**: Adds control tokens at the beginning of the sequence
**add**: Directly modifies the latent space or HVO features
**compact_attention**: Controls influence all sequence positions through attention
**self_attention**: Controls learn inter-dependencies before applying their effects

_Understanding Control Types_

Discrete controls: Integer values representing categorical choices (e.g., genre, style)

Range: [0, n_tokens-1]
Example: 5.0f represents token index 5


Continuous controls: Float values representing continuous parameters (e.g., tempo, energy)

Range: [0.0, 1.0]
Example: 0.7f represents 70% of the parameter range



8. Advanced Usage Patterns

Latent Space Interpolation

```cpp

// Encode two different patterns
auto latent1 = model.get_method("encodeLatent")({pattern1, enc_controls});
auto latent2 = model.get_method("encodeLatent")({pattern2, enc_controls});

torch::Tensor z1 = latent1.toTuple()->elements()[2].toTensor();
torch::Tensor z2 = latent2.toTuple()->elements()[2].toTensor();

// Interpolate between latents
float alpha = 0.5f; // Interpolation factor [0, 1]
torch::Tensor z_interp = (1.0f - alpha) * z1 + alpha * z2;

// Decode interpolated latent
auto decoded = model.get_method("decode")({z_interp, decoding_controls});

```

Control Sweeping

```cpp
// Sweep through continuous control values
for (float control_val = 0.0f; control_val <= 1.0f; control_val += 0.1f) {
    std::vector<float> controls = {control_val, 0.5f}; // Vary first control
    torch::Tensor dec_controls = torch::tensor(controls, torch::kFloat).reshape({1, 2});
    
    auto output = model.get_method("predict")({hvo_input, enc_controls, dec_controls});
    // Process output...
}
```

Random Generation

```cpp
// Generate random latent vector
int latent_dim = 16; // Get from model configuration
torch::Tensor random_latent = torch::randn({1, latent_dim});

// Random continuous controls
torch::Tensor random_dec_controls = torch::rand({1, num_dec_controls});

auto generated = model.get_method("decode")({random_latent, random_dec_controls});
```


9. Performance Optimization Tips

10. Memory Management
```
cpp// Disable gradient computation for inference
torch::NoGradGuard no_grad;

// Use CPU for small models, GPU for large models
torch::Device device(torch::kCPU);
// torch::Device device(torch::kCUDA, 0); // For GPU

model.to(device);
hvo_input = hvo_input.to(device);
encoding_controls = encoding_controls.to(device);
decoding_controls = decoding_controls.to(device);
```

Batch Processing (if needed)

```cpp
// Process multiple patterns at once
torch::Tensor batch_hvo = torch::rand({batch_size, 32, 3});
torch::Tensor batch_enc_controls = torch::rand({batch_size, n_enc_controls});
torch::Tensor batch_dec_controls = torch::rand({batch_size, n_dec_controls});

// Same inference calls work with batched inputs
auto batch_output = model.get_method("predict")({batch_hvo, batch_enc_controls, batch_dec_controls});
```