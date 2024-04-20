---
icon: material/notebook-multiple
---

# My Projects Index

<style>
  #my-projects-index {
    display: none;
  }
</style>

{% for project in projects_index.projects %}

<div class="grid cards" markdown >

  - ## {% if project.completed_on %} :material-book-check:{ .hover-icon-bounce .success-hover title="{{ project.completed_on }}" } {% else %} :material-book-sync:{ .hover-icon-bounce .warning-hover title="Currently Working" } {% endif %} **{{ project.title }}**

    {{ project.description }}{ style="text-align: justify;" }

    {% for point in project.extra_desc %}
    - {{ point }}
    {% endfor %}

    ---

    <p align=center markdown>
    {% for url in project.urls %}[:{{ url.icon }}:{ .light .secondary-hover }]({{ url.url }}){ target=blank_ title="{{ url.title }}" } &nbsp; &nbsp; {% endfor %}
    :material-slash-forward: &nbsp; &nbsp;
    {% for tag in project.tags|sort %} :simple-{{ tag }}:{ .{{ tag }} .hover-icon-bounce title="{{ tag|title }}"} &nbsp; {% endfor %}
    </p>

</div>

{% endfor %}

<p align=center markdown>
[:octicons-repo-16: Other Projects](others.md){ .md-button .md-button--primary .slim-button }
</p>
