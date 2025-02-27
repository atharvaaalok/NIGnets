# Neural Injective Geometry
Neural Injective Geometry networks (NIGnets) for non-self-intersecting geometry.

<div align="center">
    <img alt="NIGnets Logo with text on the side." src="https://github.com/atharvaaalok/NIGnets/blob/main/docs/assets/logos/logo_with_text_inside.svg" height="300px">
</div>


## Project Plan
- [x] Add .gitignore for the project.
- [x] Create first cut documentation pages using Jupyterbooks and MyST markdown.
    - [x] Motivation for non-self-intersecting geometry.
    - [x] Add .gitignore for MyST markdown.
    - [x] Launch web page for documentation using github pages.
- [x] Create Injective Networks.
    - [x] Basic architecture.
    - [x] Generate proper documentation.
        - [x] Proper docstrings. Follow Google python coding style guide and numpy style guide.
        - [x] Use math equations.
        - [x] Use type annotations.
    - [x] Impossible intersection using matrix exponential.
- [x] Use geosimilarity for loss functions.
- [x] Add testing code.
    - [x] Create automated training function.
    - [x] Create plot function. Parameterized and target shape comparison.
    - [x] Generate a bunch of target shapes. Use SVGs.
- [x] Update documentation with Injective Networks and showcase.
    - [x] Add documentation on Injective Networks.
    - [x] Fit basic shapes using Injective Networks.
    - [x] Create showcase for Injective Networks.
    - [x] Create showcase for Injective Networks with impossible intersection.
- [x] Add license.
- [x] Create logos.
    - [x] Create logo.
    - [x] Create favicon.
    - [x] Use on website.
- [x] Add Monotonic networks.
    - [x] Add Min-Max nets.
    - [x] Add Smooth Min-Max nets.
    - [x] Add documentation for Monotonic Nets.
    - [x] Add showcase for Monotonic Nets.
- [ ] Create ResNet-like architecture using skip connections.
    - [ ] Add skip connections that preserve injectivity.
    - [ ] Add showcase for ResNet architecture.
- [ ] Add Auxilliary networks.
    - [x] Add Pre-Aux nets.
    - [x] Add Post-Aux nets.
    - [x] Add documentation for Aux Nets.
    - [ ] Add showcase for Pre and Post nets separately and combined.
- [ ] Fit repeating and fractal shapes.
    - [ ] Use trignometric activations in Pre-Aux networks.
- [ ] 3D NIGnets.
    - [ ] Create documentation for 3D NIGnets.
    - [ ] Create 3D surface point clouds to fit to.
    - [ ] Fit 3D geometric shapes for showcase.
- [ ] Experiment with different geometric loss functions.