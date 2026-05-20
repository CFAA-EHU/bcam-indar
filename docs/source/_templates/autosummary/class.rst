{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members:

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
