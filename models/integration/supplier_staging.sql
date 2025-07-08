{{ config(materialized='view') }}
select *
from {{ source('integration', 'Supplier_Staging') }}
