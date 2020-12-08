

RESULTS_TEMPLATE_TABLE="""
</p>
<div data-hs-responsive-table="true" style="overflow-x: auto; max-width: 100%; width: 100%; margin-left: auto; margin-right: auto;">
<table border="1" cellpadding="4" style="width: 100%; border-color: #99acc2; border-style: solid; border-collapse: collapse; table-layout: fixed; height: 211px;">
<tbody>
<tr style="height: 39px;">
<td style="width: 14.5833%; height: 39px;">Time series</td>
<td style="width: 13.4583%; height: 39px;">
<p>Training error</p>
</td>
<td style="width: 14.3541%; height: 39px;">Prediction error</td>
<td style="width: 18.4896%; height: 39px;">Robust training error</td>
<td style="width: 18.4948%; height: 39px;">Robust prediction error</td>
<td style="width: 20.6198%; height: 39px;">Percentage change in prediction error</td>
</tr>
ALLROWS
</tbody>
</table>
</div>
<p>&nbsp;</p>
<p>&nbsp;</p>
"""

RESULTS_TEMPLATE_ROW_GREEN = """
tr style="height: 43px;">
<td style="width: 14.5833%; height: 43px;"><a href="https://www.microprediction.org/stream_dashboard.html?stream=NAME" rel="noopener">SHORTIE</a></td>
<td style="width: 13.4583%; height: 43px;">IN_SAMPLE_BEFORE</td>
<td style="width: 14.3541%; height: 43px;">OUT_SAMPLE_BEFORE</td>
<td style="width: 18.4896%; height: 43px;">IN_SAMPLE_ROBUST</td>
<td style="width: 18.4948%; height: 43px;">OUT_SAMPLE_ROBUST</td>
<td style="width: 20.6198%; height: 43px;"><span style="color: #000000; background-color: #00ff03;">PERCENT</span></td>
</tr>
"""

RESULTS_TEMPLATE_ROW_RED = RESULTS_TEMPLATE_ROW_GREEN.replace('#00ff03','#ea9999')


def short_name(name):
    return name.split('-')[0].replace('.json','')

def make_row(errs, name, ndigits=3):
    """
    :param errs: [ [ float ] ]   Lists in-sample/out-sample then same for robust
    :return:
    """

    in_sample  = round( errs[0][0], ndigits=ndigits )
    out_sample = round( errs[0][1], ndigits=ndigits )
    in_sample_robust = round(errs[1][0], ndigits=ndigits )
    out_sample_robust = round(errs[1][1], ndigits=ndigits )
    ratio = errs[1][1]/errs[0][1]
    good = ratio<1
    prc  = round(100*(ratio-1),1)
    replacements = {'IN_SAMPLE_BEFORE':in_sample,
                    'OUT_SAMPLE_BEFORE':out_sample,
                    'IN_SAMPLE_ROBUST':in_sample_robust,
                    'OUT_SAMPLE_ROBUST':out_sample_robust,
                    'PERCENT':prc,
                    'NAME':name.replace('.json',''),
                    'SHORTIE':short_name(name)}
    row = RESULTS_TEMPLATE_ROW_GREEN if good else RESULTS_TEMPLATE_ROW_RED
    for k,v in replacements.items():
        row = row.replace(k,str(v))
    return row


if __name__=='__main__':
    errs = [[3.0,4.0 ],[3.1,3.6] ]
    row1 = make_row(errs=errs, name='electricity-fueltype-nyiso-wind.json')
    print(row1)