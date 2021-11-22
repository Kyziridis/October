from pydantic import BaseModel, ValidationError, validator
from typing import Optional
from typing_extensions import Literal

# constants
VALUE_1 = "1-9"
VALUE_2 = "10-99"
VALUE_3 = "99+"
VALUE_4 = "unknown"


class Company(BaseModel):
    name: str
    employees: Optional[Literal[VALUE_1, VALUE_2, VALUE_3, VALUE_4]]

    @validator('employees', pre=True)
    def employees_check(cls, v):
        allowd_set = {VALUE_1, VALUE_2, VALUE_3, VALUE_4}
        if v not in allowd_set:
            if int(v) in range(1, 10):
                v = VALUE_1
            elif int(v) in range(10, 100):
                v = VALUE_2
            elif int(v) > 99:
                v = VALUE_3
        return v


if __name__ == '__main__':
    data = {'name': 'Good Company B.V.', 'employees': '100'}
    try:
        company = Company(**data)
        print(f"{company.name} has {company.employees} number of employees")
    except ValidationError:
        print(f"Invalid data supplied")
        raise
