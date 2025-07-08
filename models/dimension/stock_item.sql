{{ config(materialized='view') }}
select *
from {{ source('dimension', 'Stock Item') }}
