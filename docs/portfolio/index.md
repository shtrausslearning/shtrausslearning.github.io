---
icon: material/notebook-multiple
title: Projects
comments: true
---

A collection of **data science** projects which utilise **machine** and/or **deep learning**, I've grouped them based on the most relevant topic for your conveniene.

<div class="grid cards" markdown>

{% for project in other_projects.projects %}

  - :simple-{{ project.icon }}:{ .lg .{{ project.icon }} title="Status: {{ project.status|title }}" } &nbsp; **{{ project.title }}**{ title="Status: {{ project.status|title }}" }

    ---

    {{ project.description }}

    <br>
    <p align=center markdown>
    [:octicons-repo-16: View Section]({{ project.github }}){ .md-button .md-button--primary .slim-button }
    </p>

{% endfor %}

</div>
