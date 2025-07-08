{{ config(materialized='view') }}
select *
from {{ source('dimension', 'Payment Method') }}
