from engineering_exercise.data_model_exercise import Company


class TestDataModel:
    data = {'name': 'Good Company B.V.', 'employees': '0'}

    def test_first_bin(self):
        data = {'name': 'Good Company B.V.', 'employees': '1'}
        company = Company(**data)
        assert company.name == data.get('name')
        assert company.employees == "1-9"

    def test_second_bin(self):
        data = {'name': 'Good Company B.V.', 'employees': '10'}
        company = Company(**data)
        assert company.name == data.get('name')
        assert company.employees == "10-99"

    def test_third_bin(self):
        data = {'name': 'Good Company B.V.', 'employees': '100'}
        company = Company(**data)
        assert company.name == data.get('name')
        assert company.employees == "99+"

    def test_fourth_bin(self):
        data = {'name': 'Good Company B.V.', 'employees': 'unknown'}
        company = Company(**data)
        assert company.name == data.get('name')
        assert company.employees == "unknown"
