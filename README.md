# DLBlueprint

DLBlueprint offers a robust and flexible framework for kickstarting deep learning projects, streamlining the development process from conception to deployment. Designed with scalability and ease of integration in mind, this repository serves as an essential template for both beginners and experienced developers. Featuring pre-configured modules for common deep learning tasks, DLBlueprint leverages popular libraries and tools to ensure your projects are built on a solid foundation. Whether you're working on computer vision, natural language processing, or any other AI-driven application, DLBlueprint accelerates your development cycle, allowing you to focus on innovation and research. Dive into our comprehensive guides and examples to see how DLBlueprint can transform your next project.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [CMakeLists.txt](#cmakelists)
- [Example YAML Configuration](#example-yaml-configuration)
- [C++ Test Example](#c-test-example)

## Installation
To install DLBlueprint, follow these steps:
1. Clone the repository to your local machine.
2. Ensure you have the necessary dependencies installed, including:
   - `CMake` version 3.17 or higher
   - `fmt` library
   - `Boost` library
   - `yaml-cpp` library
   - `tl-ranges` library
   - `libtorch` (PyTorch C++ library)
3. Install all dependencies with vcpkg (libtorch requires you to download seperately)
4. Compile the project using CMake.

## Usage
To use DLBlueprint, follow these steps:
1. Create a YAML configuration file specifying the architecture of your deep learning model.
2. Load the YAML file in your C++ code using the `YAML::LoadFile` function.
3. Build the model using `dlb::Build` function, passing the input dimension and the YAML configuration.
4. Reset the state of the model if necessary.
5. Access the compiled modules and perform forward pass using the generated blueprint.


## Example YAML Configuration
```yaml
shared:
  <modules>:
    output:GRU:
      hidden_size: 128
      return_all_seq: false

  <children>:
    actor:
      <modules>:
        feature:FCNN:
          dims: [ 64, 64]
          activations: tanh

        action:Linear:
          in_features: 2

    critic:
      <modules>:
        output:FCNN:
          dims: [ 64, 32 ]
          activations: tanh

        value:Linear:
          in_features: 1

model:
  <modules>:
    return:FCNN:
      dims: [ 64, 64]
      activations: tanh
      weight_init:
        type: orthogonal
        gain: 1.414213562
      bias_init: 
        type: constant
        gain: 0
      new_bias: true
```

## Contributing
If you'd like to contribute to DLBlueprint, please follow the guidelines outlined in the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support
If you encounter any issues or have questions about DLBlueprint, please [open an issue](https://github.com/yourusername/dlblueprint/issues) on GitHub or reach out to the project maintainers.

## Roadmap
We're constantly improving DLBlueprint and adding new features. Check out our [roadmap](ROADMAP.md) to see what's coming next!

## Acknowledgements
We would like to thank all contributors to DLBlueprint for their valuable contributions.

## Conclusion
Thank you for using DLBlueprint! We hope it accelerates your deep learning projects and simplifies your development process. If you find it useful, please consider starring the repository on GitHub and spreading the word. Happy coding! 🚀
