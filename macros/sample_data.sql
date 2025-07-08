{% macro sample_data(model_name, limit=5) %}
  {{ run_query("SELECT TOP {{ limit }} * FROM " ~ ref(model_name)) }}
{% endmacro %}
