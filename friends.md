---
icon: fontawesome/solid/users-viewfinder
hide:
  - navigation
  - toc
  - feedback
---

<style>
  article > h1 { display: none; }
  @media (min-width: 900px) {
    main > div > div.md-content {
      max-width: 75%;
      margin: auto;
    }
  }
</style>

<div class="grid cards" markdown>

{% for friend in friends.friends %}

  - <p align="center">[![avatar]({{ friend.avatar }}){ .twemoji .xxxl .base-border-radius }]({{ friend.primaryUrl }})</p>

    <p align="center">**{{ friend.name }}** • <span class="secondary">{{ friend.profession }}</span></p>

    ---

    <p align="justify">{{ friend.description }}</p>

    ---

    <p align="center">{% for social in friend.socials | sort(attribute="title") %} [:{{ social.icon | replace("/", "-") }}:{ .lg .light .hover-icon-bounce title="{{ social.title }}" }]({{ social.url }}) &nbsp; {% endfor %}</p>

{% endfor %}

</div>
