# Personal Website

<p align="center">
    <img src="data/assets/home.png" alt="arv-anshul"/>
</p>

<p align="center">
    <a href="https://squidfunk.github.io/mkdocs-material/"><img src="https://img.shields.io/badge/Material_for_MkDocs-526CFE?style=for-the-badge&amp;logo=MaterialForMkDocs&amp;logoColor=white" alt="Built with Material for MkDocs"></a>
    <a href="https://arv-anshul.github.io"><img src="https://img.shields.io/badge/GitHub%20Pages-222?logo=github&logoColor=fff&style=for-the-badge" alt="GitHub Badge"></a>
    <a href="https://github.com/arv-anshul/arv-anshul.github.io/actions"><img src="https://img.shields.io/badge/GitHub%20Actions-2088FF?logo=githubactions&logoColor=fff&style=for-the-badge" alt="GitHub Actions Badge"></a>
</p>

## Website's Features

- **Portfolio**: Showcase of my skills, achievements, and projects to provide an overview of my professional journey.
- **Project Index**: A comprehensive index highlighting the details of various projects I've undertaken, including descriptions, technologies used, and outcomes.
- **Projects Details**: In-depth information and specifications about each project, offering a closer look into the methodologies, challenges, and solutions implemented during the development process.
- **Blogs**: A collection of insightful and informative blogs covering a range of topics, sharing knowledge and experiences with the audience.

## Setup Locally

Clone this repository with `git`:

```bash
git clone https://github.com/arv-anshul/arv-anshul.github.io.git
```

### Install with `rye`

Install `rye` from [official website](https://rye-up.com) and create virtual environment using `rye`.

```bash
rye sync --no-dev
```

Now, you can easily serve the `docs/` folder as a website:

```bash
rye mkdocs serve
```

### Install with `pip`

Install dependencies from `requirements.lock` (works as `requirements.txt`) file:

```bash
pip install -r requirements.lock
```

Now, you can easily serve the `docs/` folder as a website:

```bash
mkdocs serve
```

## Acknowledgement

1. A very big thanks to [@squidfunk](https://github.com/squidfunk/mkdocs-material) for making `mkdocs` this easy to use and maintain.
2. A pulgin `mkdocs-markdownextradata-plugin` which help to render the `.md` with `jinja` template. Thanks to [@rosscdh](https://github.com/rosscdh/mkdocs-markdownextradata-plugin/) to create this pulgin.
3. (maybe) My first interaction with `mkdocs` happend while learning [FastAPI](https://fastapi.tiangolo.com/) from its amazing documentation.
4. Finally [@me](https://github.com/arv-anshul), I have learned this tool in almost one day and **I am writting this line at `02:10 AM` midnight in a very cold night and also listening song "Maine Poochha Chand Se"**.
