#!pip install faker
#!pip install kudu

from faker import Faker
import kudu, time, random

fake = Faker()

kuduClient = kudu.connect(host='10.0.0.25', port=7051)
kuduSession = kuduClient.new_session()
kuduTable = kuduClient.table('impala::network.customers')

fake.seed(242)
  
for i in range(1,1000001):

    c = {}
    c['id'] = i
    c['name'] = fake.name()
    c['zipcode'] = fake.zipcode()
    c['state'] = fake.state()
    c['age'] = int(random.normalvariate(40, 10))
    c['sex'] = fake.random_element(elements=('m', 'f'))
    c['hh_members'] = int(random.normalvariate(3, 1))
    c['services'] = int(random.normalvariate(3, 1))
    c['income'] = int(random.normalvariate(80000, 15000))
    member_since = fake.date_between(start_date="-10y", end_date="-1y")
    c['cust_since'] = long(time.mktime(member_since.timetuple()))
    c['avg_monthly_data'] = int(random.normalvariate(300000000, 50000000))
    c['last_month_data'] = int(random.normalvariate(300000000, 50000000))
    c['avg_monthly_watch'] = int(random.normalvariate(500000000, 30000000))
    c['last_month_watch'] = int(random.normalvariate(500000000, 30000000))
    c['monthly_spend'] = int(random.normalvariate(10, 2)) + \
                          c['age']/4 + \
                          int(c['sex']=='m')*10 + \
                          c['income']/8000 + \
                          c['avg_monthly_data']/30000000
    
    insOp = kuduTable.new_insert(c)
    kuduSession.apply(insOp)

    if not i % 10000:
      try:
          kuduSession.flush()
          print('Wrote %d records'%i)
      except kudu.KuduBadStatus as e:
          print(kuduSession.get_pending_errors())