#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <unistd.h>
#include <sys/wait.h>
#include <pthread.h>
#include <semaphore.h>
#include <cstring>
#include <cmath>

using namespace std;

// Global configuration
int num_hidden_layers;
int neurons_per_layer;
double** all_weights;  // Array of pointers to weight arrays
int* weight_sizes;      // Size of each weight array
double* input_values;  // Array of 2 input values

// Structure for neuron thread data
struct NeuronData {
    int layer_id;
    int neuron_id;
    double* inputs;
    int num_inputs;
    double* weights;
    int num_weights;
    double output;
    pthread_mutex_t* mutex;
};

// Helper function to parse comma-separated values from a line
int parseLine(ifstream& file, double* values, int max_values) {
    string line;
    if (!getline(file, line)) {
        return 0;
    }
    
    // Remove any trailing whitespace
    while (!line.empty() && (line.back() == ' ' || line.back() == '\t' || line.back() == '\r')) {
        line.pop_back();
    }
    
    if (line.empty()) {
        return 0;
    }
    
    int count = 0;
    stringstream ss(line);
    string token;
    
    while (getline(ss, token, ',') && count < max_values) {
        // Trim whitespace from token
        size_t start = token.find_first_not_of(" \t\r");
        size_t end = token.find_last_not_of(" \t\r");
        if (start != string::npos && end != string::npos) {
            token = token.substr(start, end - start + 1);
            values[count] = stod(token);
            count++;
        }
    }
    
    return count;
}

// Function to read input file
bool readInputFile(const char* filename, int num_hidden, int neurons_per) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open input.txt" << endl;
        return false;
    }

    // Allocate memory for input values (always 2)
    input_values = new double[2];
    double temp_values[10];
    int count = parseLine(file, temp_values, 10);
    if (count < 2) {
        cerr << "Error: Not enough input values" << endl;
        file.close();
        return false;
    }
    input_values[0] = temp_values[0];
    input_values[1] = temp_values[1];

    // Total layers: 1 input + num_hidden + 1 output
    int total_layers = num_hidden + 2;
    all_weights = new double*[total_layers];
    weight_sizes = new int[total_layers];

    // Read weights for input layer (2 neurons, each with neurons_per weights)
    // Each input neuron connects to all neurons in the first hidden layer
    int input_layer_size = 2 * neurons_per;
    all_weights[0] = new double[input_layer_size];
    weight_sizes[0] = input_layer_size;
    
    double line_values[10];
    int idx = 0;
    // Read weights for input layer (2 neurons × neurons_per weights each)
    while (idx < input_layer_size) {
        count = parseLine(file, line_values, 10);
        for (int i = 0; i < count && idx < input_layer_size; i++) {
            all_weights[0][idx++] = line_values[i];
        }
    }

    // Read weights for hidden layers
    for (int layer = 0; layer < num_hidden; layer++) {
        int num_inputs = (layer == 0) ? 2 : neurons_per;
        int layer_size = neurons_per * num_inputs;
        all_weights[layer + 1] = new double[layer_size];
        weight_sizes[layer + 1] = layer_size;
        
        idx = 0;
        // Read enough lines to get all weights for this layer
        while (idx < layer_size) {
            count = parseLine(file, line_values, 10);
            for (int i = 0; i < count && idx < layer_size; i++) {
                all_weights[layer + 1][idx++] = line_values[i];
            }
        }
    }

    // Read weights for output layer
    int output_layer_size = neurons_per * neurons_per;
    all_weights[total_layers - 1] = new double[output_layer_size];
    weight_sizes[total_layers - 1] = output_layer_size;
    
    idx = 0;
    // Read enough lines to get all weights for output layer
    while (idx < output_layer_size) {
        count = parseLine(file, line_values, 10);
        for (int i = 0; i < count && idx < output_layer_size; i++) {
            all_weights[total_layers - 1][idx++] = line_values[i];
        }
    }

    file.close();
    return true;
}

// Neuron thread function
void* neuronThread(void* arg) {
    NeuronData* data = (NeuronData*)arg;
    
    // Compute weighted sum
    double sum = 0.0;
    int min_size = (data->num_inputs < data->num_weights) ? data->num_inputs : data->num_weights;
    for (int i = 0; i < min_size; i++) {
        sum += data->inputs[i] * data->weights[i];
    }
    
    data->output = sum;
    
    if (data->mutex) {
        pthread_mutex_lock(data->mutex);
        cout << "Layer " << data->layer_id << ", Neuron " << data->neuron_id 
             << ": Output = " << data->output << endl;
        pthread_mutex_unlock(data->mutex);
    }
    
    return nullptr;
}

// Input layer process
void inputLayerProcess(int forward_pipe_write, int backward_pipe_read, 
                       double* initial_inputs, 
                       double* weights, int weight_size,
                       pthread_mutex_t* output_mutex, ofstream* out_file) {
    cout << "Input Layer Process Started (PID: " << getpid() << ")" << endl;
    
    double* current_inputs = new double[2];
    current_inputs[0] = initial_inputs[0];
    current_inputs[1] = initial_inputs[1];
    
    // First forward pass
    cout << "=== First Forward Pass ===" << endl;
    cout << "Input Layer: Initial inputs: " << current_inputs[0] << ", " << current_inputs[1] << endl;
    
    // Create threads for 2 neurons
    pthread_t threads[2];
    NeuronData neuron_data[2];
    pthread_mutex_t local_mutex = PTHREAD_MUTEX_INITIALIZER;
    
    // Input layer neurons: each takes 2 inputs and has neurons_per weights
    // Each input neuron uses the 2 input values with its weights
    // The output is computed as: sum(input_i × weight_i) for i=0 to 1
    // (using first 2 weights since we have 2 inputs)
    int num_weights_per_neuron = weight_size / 2; // Total weights / 2 neurons
    
    for (int i = 0; i < 2; i++) {
        neuron_data[i].layer_id = 0;
        neuron_data[i].neuron_id = i;
        neuron_data[i].inputs = current_inputs;
        neuron_data[i].num_inputs = 2;
        // Each input neuron has num_weights_per_neuron weights, but uses first 2
        neuron_data[i].weights = new double[2]; // Only use first 2 weights
        for (int j = 0; j < 2; j++) {
            neuron_data[i].weights[j] = weights[i * num_weights_per_neuron + j];
        }
        neuron_data[i].num_weights = 2;
        neuron_data[i].mutex = &local_mutex;
        pthread_create(&threads[i], nullptr, neuronThread, &neuron_data[i]);
    }
    
    // Wait for all threads
    double* outputs = new double[2];
    for (int i = 0; i < 2; i++) {
        pthread_join(threads[i], nullptr);
        outputs[i] = neuron_data[i].output;
    }
    
    // Write outputs to pipe (2 values from 2 input neurons)
    write(forward_pipe_write, outputs, sizeof(double) * 2);
    close(forward_pipe_write);
    
    pthread_mutex_lock(output_mutex);
    *out_file << "========================================" << endl;
    *out_file << "FIRST FORWARD PASS" << endl;
    *out_file << "========================================" << endl;
    *out_file << "Input Layer:" << endl;
    *out_file << "  Neuron 0: " << outputs[0] << endl;
    *out_file << "  Neuron 1: " << outputs[1] << endl;
    *out_file << endl;
    pthread_mutex_unlock(output_mutex);
    
    cout << "Input Layer: Forward pass completed. Outputs sent: " 
         << outputs[0] << ", " << outputs[1] << endl;
    
    // Wait for backward signal
    double* backward_inputs = new double[2];
    read(backward_pipe_read, backward_inputs, sizeof(double) * 2);
    close(backward_pipe_read);
    
    cout << "Input Layer: Received backward signals: " 
         << backward_inputs[0] << ", " << backward_inputs[1] << endl;
    
    pthread_mutex_lock(output_mutex);
    *out_file << "Input Layer received:" << endl;
    *out_file << "  f(x1): " << backward_inputs[0] << endl;
    *out_file << "  f(x2): " << backward_inputs[1] << endl;
    *out_file << endl;
    *out_file << "========================================" << endl;
    *out_file << "SECOND FORWARD PASS" << endl;
    *out_file << "========================================" << endl;
    pthread_mutex_unlock(output_mutex);
    
    // Second forward pass using backward outputs as new inputs
    cout << "\n=== Second Forward Pass ===" << endl;
    cout << "Input Layer: Using backward outputs as new inputs: " 
         << backward_inputs[0] << ", " << backward_inputs[1] << endl;
    
    // Update inputs for second pass
    current_inputs[0] = backward_inputs[0];
    current_inputs[1] = backward_inputs[1];
    
    // Create new threads for second forward pass
    for (int i = 0; i < 2; i++) {
        neuron_data[i].inputs = current_inputs;
        pthread_create(&threads[i], nullptr, neuronThread, &neuron_data[i]);
    }
    
    for (int i = 0; i < 2; i++) {
        pthread_join(threads[i], nullptr);
        outputs[i] = neuron_data[i].output;
    }
    
    pthread_mutex_lock(output_mutex);
    *out_file << "Input Layer:" << endl;
    *out_file << "  Neuron 0: " << outputs[0] << endl;
    *out_file << "  Neuron 1: " << outputs[1] << endl;
    *out_file << endl;
    *out_file << "========================================" << endl;
    pthread_mutex_unlock(output_mutex);
    
    cout << "Input Layer: Second forward pass completed. Outputs: " 
         << outputs[0] << ", " << outputs[1] << endl;
    
    // Cleanup
    for (int i = 0; i < 2; i++) {
        delete[] neuron_data[i].weights;
    }
    delete[] current_inputs;
    delete[] outputs;
    delete[] backward_inputs;
    pthread_mutex_destroy(&local_mutex);
    exit(0);
}

// Hidden layer process
void hiddenLayerProcess(int layer_id, int num_hidden, int neurons_per, 
                        int forward_pipe_read, int forward_pipe_write,
                        int backward_pipe_read, int backward_pipe_write,
                        double* weights, int weight_size,
                        pthread_mutex_t* output_mutex, ofstream* out_file) {
    cout << "Hidden Layer " << layer_id << " Process Started (PID: " << getpid() << ")" << endl;
    
    // Read inputs from previous layer
    int num_inputs = (layer_id == 1) ? 2 : neurons_per;
    double* inputs = new double[num_inputs];
    read(forward_pipe_read, inputs, sizeof(double) * num_inputs);
    close(forward_pipe_read);
    
    cout << "Hidden Layer " << layer_id << ": Received inputs from previous layer." << endl;
    
    // Create threads for neurons
    pthread_t* thread_array = new pthread_t[neurons_per];
    NeuronData* neuron_data = new NeuronData[neurons_per];
    pthread_mutex_t local_mutex = PTHREAD_MUTEX_INITIALIZER;
    double* outputs = new double[neurons_per];
    
    for (int i = 0; i < neurons_per; i++) {
        neuron_data[i].layer_id = layer_id;
        neuron_data[i].neuron_id = i;
        neuron_data[i].inputs = inputs;
        neuron_data[i].num_inputs = num_inputs;
        neuron_data[i].weights = new double[num_inputs];
        for (int j = 0; j < num_inputs; j++) {
            neuron_data[i].weights[j] = weights[i * num_inputs + j];
        }
        neuron_data[i].num_weights = num_inputs;
        neuron_data[i].mutex = &local_mutex;
        pthread_create(&thread_array[i], nullptr, neuronThread, &neuron_data[i]);
    }
    
    // Wait for all threads
    for (int i = 0; i < neurons_per; i++) {
        pthread_join(thread_array[i], nullptr);
        outputs[i] = neuron_data[i].output;
    }
    
    // Write outputs to next layer
    write(forward_pipe_write, outputs, sizeof(double) * neurons_per);
    close(forward_pipe_write);
    
    pthread_mutex_lock(output_mutex);
    *out_file << "Hidden Layer " << layer_id << ":" << endl;
    for (int i = 0; i < neurons_per; i++) {
        *out_file << "  Neuron " << i << ": " << outputs[i] << endl;
    }
    *out_file << endl;
    pthread_mutex_unlock(output_mutex);
    
    cout << "Hidden Layer " << layer_id << ": Forward pass completed. Outputs sent to next layer." << endl;
    
    // Handle backward pass
    // All layers receive 2 values (f(x1), f(x2))
    double* backward_inputs = new double[2];
    read(backward_pipe_read, backward_inputs, sizeof(double) * 2);
    close(backward_pipe_read);
    
    cout << "Hidden Layer " << layer_id << ": Received backward signals: "
         << backward_inputs[0] << ", " << backward_inputs[1] << endl;
    
    pthread_mutex_lock(output_mutex);
    *out_file << "Hidden Layer " << layer_id << " received:" << endl;
    *out_file << "  f(x1): " << backward_inputs[0] << endl;
    *out_file << "  f(x2): " << backward_inputs[1] << endl;
    *out_file << endl;
    pthread_mutex_unlock(output_mutex);
    
    // Send backward to previous layer (always 2 values: f(x1) and f(x2))
    double* backward_outputs = new double[2];
    backward_outputs[0] = backward_inputs[0];
    backward_outputs[1] = backward_inputs[1];
    write(backward_pipe_write, backward_outputs, sizeof(double) * 2);
    close(backward_pipe_write);
    
    // Cleanup
    for (int i = 0; i < neurons_per; i++) {
        delete[] neuron_data[i].weights;
    }
    delete[] thread_array;
    delete[] neuron_data;
    delete[] inputs;
    delete[] outputs;
    delete[] backward_inputs;
    delete[] backward_outputs;
    pthread_mutex_destroy(&local_mutex);
    
    exit(0);
}

// Output layer process
void outputLayerProcess(int forward_pipe_read, int backward_pipe_write,
                        int neurons_per, double* weights, int weight_size,
                        pthread_mutex_t* output_mutex, ofstream* out_file) {
    cout << "Output Layer Process Started (PID: " << getpid() << ")" << endl;
    
    // Read inputs from previous layer
    double* inputs = new double[neurons_per];
    read(forward_pipe_read, inputs, sizeof(double) * neurons_per);
    close(forward_pipe_read);
    
    cout << "Output Layer: Received inputs from previous layer." << endl;
    
    // Create threads for neurons
    pthread_t* thread_array = new pthread_t[neurons_per];
    NeuronData* neuron_data = new NeuronData[neurons_per];
    pthread_mutex_t local_mutex = PTHREAD_MUTEX_INITIALIZER;
    double* outputs = new double[neurons_per];
    
    for (int i = 0; i < neurons_per; i++) {
        neuron_data[i].layer_id = -1; // Output layer
        neuron_data[i].neuron_id = i;
        neuron_data[i].inputs = inputs;
        neuron_data[i].num_inputs = neurons_per;
        neuron_data[i].weights = new double[neurons_per];
        for (int j = 0; j < neurons_per; j++) {
            neuron_data[i].weights[j] = weights[i * neurons_per + j];
        }
        neuron_data[i].num_weights = neurons_per;
        neuron_data[i].mutex = &local_mutex;
        pthread_create(&thread_array[i], nullptr, neuronThread, &neuron_data[i]);
    }
    
    // Wait for all threads
    for (int i = 0; i < neurons_per; i++) {
        pthread_join(thread_array[i], nullptr);
        outputs[i] = neuron_data[i].output;
    }
    
    // Compute sum of outputs
    double output_sum = 0.0;
    for (int i = 0; i < neurons_per; i++) {
        output_sum += outputs[i];
    }
    
    // Compute f(x1) and f(x2)
    double fx1 = (output_sum * output_sum + output_sum + 1) / 2.0;
    double fx2 = (output_sum * output_sum - output_sum) / 2.0;
    
    cout << "Output Layer: Sum = " << output_sum << ", f(x1) = " << fx1 << ", f(x2) = " << fx2 << endl;
    
    // Write to output file
    pthread_mutex_lock(output_mutex);
    *out_file << "Output Layer:" << endl;
    for (int i = 0; i < neurons_per; i++) {
        *out_file << "  Neuron " << i << ": " << outputs[i] << endl;
    }
    *out_file << "  Sum: " << output_sum << endl;
    *out_file << "  f(x1): " << fx1 << endl;
    *out_file << "  f(x2): " << fx2 << endl;
    *out_file << endl;
    *out_file << "========================================" << endl;
    *out_file << "BACKWARD PASS" << endl;
    *out_file << "========================================" << endl;
    pthread_mutex_unlock(output_mutex);
    
    // Send backward signals (f(x1) and f(x2))
    double* backward_signals = new double[2];
    backward_signals[0] = fx1;
    backward_signals[1] = fx2;
    write(backward_pipe_write, backward_signals, sizeof(double) * 2);
    close(backward_pipe_write);
    
    cout << "Output Layer: Backward signals sent (f(x1)=" << fx1 << ", f(x2)=" << fx2 << ")" << endl;
    
    // Cleanup
    for (int i = 0; i < neurons_per; i++) {
        delete[] neuron_data[i].weights;
    }
    delete[] thread_array;
    delete[] neuron_data;
    delete[] inputs;
    delete[] outputs;
    delete[] backward_signals;
    pthread_mutex_destroy(&local_mutex);
    
    exit(0);
}

int main() {
    cout << "=== Multi-Core Neural Network Simulation ===" << endl;
    
    // Get user input
    cout << "Enter number of hidden layers: ";
    cin >> num_hidden_layers;
    cout << "Enter number of neurons per layer: ";
    cin >> neurons_per_layer;
    
    // Read input file
    if (!readInputFile("input.txt", num_hidden_layers, neurons_per_layer)) {
        return 1;
    }
    
    // Open output file
    ofstream output_file("output.txt");
    if (!output_file.is_open()) {
        cerr << "Error: Cannot create output.txt" << endl;
        return 1;
    }
    
    // Create mutex for output file access
    pthread_mutex_t output_mutex = PTHREAD_MUTEX_INITIALIZER;
    
    // Create pipes for forward communication
    int num_pipes = num_hidden_layers + 1;
    int** forward_pipes = new int*[num_pipes];
    for (int i = 0; i < num_pipes; i++) {
        forward_pipes[i] = new int[2];
        if (pipe(forward_pipes[i]) == -1) {
            cerr << "Error creating pipe" << endl;
            return 1;
        }
    }
    
    // Create pipes for backward communication
    int** backward_pipes = new int*[num_pipes];
    for (int i = 0; i < num_pipes; i++) {
        backward_pipes[i] = new int[2];
        if (pipe(backward_pipes[i]) == -1) {
            cerr << "Error creating backward pipe" << endl;
            return 1;
        }
    }
    
    // Fork input layer process
    pid_t input_pid = fork();
    if (input_pid == 0) {
        // Child: Input layer
        close(forward_pipes[0][0]);
        close(backward_pipes[0][1]);
        inputLayerProcess(forward_pipes[0][1], backward_pipes[0][0],
                         input_values, all_weights[0], weight_sizes[0],
                         &output_mutex, &output_file);
    }
    close(forward_pipes[0][1]);
    close(backward_pipes[0][0]);
    
    // Fork hidden layer processes
    pid_t* hidden_pids = new pid_t[num_hidden_layers];
    for (int i = 0; i < num_hidden_layers; i++) {
        hidden_pids[i] = fork();
        if (hidden_pids[i] == 0) {
            // Child: Hidden layer
            close(forward_pipes[i][1]);
            close(forward_pipes[i + 1][0]);
            close(backward_pipes[i][0]);
            close(backward_pipes[i + 1][1]);
            
            hiddenLayerProcess(i + 1, num_hidden_layers, neurons_per_layer,
                             forward_pipes[i][0], forward_pipes[i + 1][1],
                             backward_pipes[i + 1][0], backward_pipes[i][1],
                             all_weights[i + 1], weight_sizes[i + 1],
                             &output_mutex, &output_file);
        }
        close(forward_pipes[i][0]);
        close(forward_pipes[i + 1][1]);
        close(backward_pipes[i][1]);
        close(backward_pipes[i + 1][0]);
    }
    
    // Fork output layer process
    pid_t output_pid = fork();
    if (output_pid == 0) {
        // Child: Output layer
        close(forward_pipes[num_hidden_layers][1]);
        close(backward_pipes[num_hidden_layers][0]);
        int output_layer_idx = num_hidden_layers + 1;
        outputLayerProcess(forward_pipes[num_hidden_layers][0],
                          backward_pipes[num_hidden_layers][1],
                          neurons_per_layer,
                          all_weights[output_layer_idx], weight_sizes[output_layer_idx],
                          &output_mutex, &output_file);
    }
    close(forward_pipes[num_hidden_layers][0]);
    close(backward_pipes[num_hidden_layers][1]);
    
    // Wait for all child processes
    waitpid(input_pid, nullptr, 0);
    for (int i = 0; i < num_hidden_layers; i++) {
        waitpid(hidden_pids[i], nullptr, 0);
    }
    waitpid(output_pid, nullptr, 0);
    
    // Close all pipes
    for (int i = 0; i < num_pipes; i++) {
        close(forward_pipes[i][0]);
        close(forward_pipes[i][1]);
        delete[] forward_pipes[i];
    }
    for (int i = 0; i < num_pipes; i++) {
        close(backward_pipes[i][0]);
        close(backward_pipes[i][1]);
        delete[] backward_pipes[i];
    }
    delete[] forward_pipes;
    delete[] backward_pipes;
    
    // Cleanup weights
    int total_layers = num_hidden_layers + 2;
    for (int i = 0; i < total_layers; i++) {
        delete[] all_weights[i];
    }
    delete[] all_weights;
    delete[] weight_sizes;
    delete[] input_values;
    delete[] hidden_pids;
    
    // Cleanup mutex
    pthread_mutex_destroy(&output_mutex);
    
    // Close output file
    output_file.close();
    
    cout << "\nSimulation completed. Results written to output.txt" << endl;
    
    return 0;
}
