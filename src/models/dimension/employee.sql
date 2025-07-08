{{ config(materialized='view') }}
select * from {{ source('dimension', 'Employee') }}
