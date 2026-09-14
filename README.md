# Dashboard NextQS e StarLed

Dashboard Streamlit que lê as planilhas operacionais das duas empresas.

A integração da conta **NextQS Brasil** coleta o mês encerrado pela Meta Marketing API e atualiza a aba `meta_campanhas` da planilha NextQS. O fluxo automático roda no GitHub Actions todo dia 2, às 00:00 no horário de São Paulo.

Consulte [a documentação da automação](docs/AUTOMACAO_META_CAMPANHAS.md) para arquitetura, campos, operação e recuperação de falhas.

## Desenvolvimento

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Os testes da integração Meta não acessam serviços externos:

```bash
python -m unittest discover -s tests -p 'test_meta_report.py' -v
```
