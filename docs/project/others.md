---
title: Other Projects
icon: octicons/repo-16
date: 2024-01-28
hide:
  - feedback
  - toc
---

# :octicons-repo-16:{ title="2024-01-28" } Other Projects

<div class="grid cards" markdown>

{% for project in other_projects.projects %}

  - :simple-{{ project.icon }}:{ .lg .{{ project.icon }} title="Status: {{ project.status|title }}" } &nbsp; **{{ project.title }}**{ title="Status: {{ project.status|title }}" }

    ---

    {{ project.description }}

    [:octicons-arrow-right-24: GitHub]({{ project.github }})

{% endfor %}

</div>
