{{ config(materialized='view') }}
select *
from {{ source('integration', 'PaymentMethod_Staging') }}
