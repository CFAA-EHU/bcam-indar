{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

	{% block methods %}
	{% if methods %}
	.. rubric:: Methods

	.. autosummary::
		:toctree: .

	{% for item in methods %}
	{% if item == '__init__' or not (item.startswith('__') and item.endswith('__')) %}
		~{{ objname }}.{{ item }}
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
		~{{ objname }}.{{ item }}
	{% endfor %}
	{% endif %}
	{% endblock %}
