{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree: .

   {% for item in methods %}
   {% if item[0] != '_' %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {% endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :toctree: .

   {% for item in attributes %}
   {% if item[0] != '_' %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {% endfor %}
   {% endif %}
   {% endblock %}
