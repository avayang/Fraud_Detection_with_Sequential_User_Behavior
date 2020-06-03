# Data Schema

## sample data
```
[
  "56f889ee11df4a72955147cb2f29a638",
  {
    "order_info": {
      "overdue": 0,
      "new_client": 0,
      "order_time": 1509322980000.0,
      "label": 0
    },
    "data": [
      {
        "pname": "loan_index",
        "pstime": 1508169825905,
        "petime": 1508169827989,
        "pid": "1508169825083X3005",
        "sid": "1508169825895"
      },
      {
        "pname": "loan_submission",
        "pstime": 1508169828161,
        "petime": 1508169832016,
        "pid": "1508169825083X3005",
        "sid": "1508169825895"
      },
      {
        "pname": "login",
        "pstime": 1509351552976,
        "petime": 1509351568401,
        "pid": "1509351523127X29603",
        "sid": "1509351523686"
      }
    ]
  }
]
```

- Data Structure
	- Each line of the text file contains the whole behavior for an individual application (Note: not an individual customer)
	- Data Type : ```[user_id, dictionary(order_info, data)] ```
- Order Info

Keys | ValueDescription | ValueType 
---|---|---
`overdue` | If application defaults, how many days overdued | int
`new_client` | Whether this user has previous `successful` application | int (1-yes / 0-no) 
`order_time`| Application time | foat from Unix time(ms)
`label` | Whether this application defaults | int (0-no / 1-yes) 

- Data
	- A list of page view behavior for an individual before a certain time (not neccessarily before application time)

Keys | ValueDescription | ValueType 
---|---|---
`pname` | Page name | String
`pstime`| Time when page viewing starts | int from Unix time(ms)
`petime`| Time when page viewing ends | int from Unix time(ms)
`pid` | Whether this application defaults | String
`sid` | Session ID | A string of int ID
