{{ config(materialized='view') }}
select *
from {{ source('dimension', 'Transaction Type') }}
