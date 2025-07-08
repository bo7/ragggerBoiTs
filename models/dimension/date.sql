{{ config(materialized='view') }}
select *
from {{ source('dimension', 'Date') }}
