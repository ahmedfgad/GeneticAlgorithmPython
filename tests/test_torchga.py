import pygad
import pygad.torchga
import torch

def test_torchga_evolution():
    """Test pygad.torchga with pygad.GA."""

    # XOR data
    data_inputs = torch.tensor([[0.0, 0.0], [0.1, 0.6], [1.0, 0.0], [1.1, 1.3]])
    data_outputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]) # One-hot encoded

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 2),
        torch.nn.Softmax(dim=1)
    )

    torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=10)

    def fitness_func(ga_instance, solution, solution_idx):
        model_weights_dict = pygad.torchga.model_weights_as_dict(model=model,
                                                                 weights_vector=solution)
        model.load_state_dict(model_weights_dict)
        predictions = model(data_inputs)
        
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(predictions, data_outputs).detach().numpy()
        fitness = 1.0 / (loss + 0.00000001)
        return fitness

    ga_instance = pygad.GA(num_generations=2,
                           num_parents_mating=5,
                           initial_population=torch_ga.population_weights,
                           fitness_func=fitness_func,
                           suppress_warnings=True)

    ga_instance.run()
    assert ga_instance.run_completed
    assert ga_instance.generations_completed == 2
    
    print("test_torchga_evolution passed.")

if __name__ == "__main__":
    test_torchga_evolution()
    print("\nAll TorchGA tests passed!")
