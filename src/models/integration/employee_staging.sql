{{ config(materialized='view') }}
select * from {{ source('integration', 'Employee_Staging') }}
