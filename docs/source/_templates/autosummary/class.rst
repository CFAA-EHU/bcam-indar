{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

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
