<?php
	$age = $_POST["age"];
        $income = $_POST["income"];
        $usage = $_POST["usage"];
        $sex = $_POST["sex"];
        $state = $_POST["state"];
        $hh_members = $_POST["hh_members"];
        $services = $_POST["services"];

	$url = 'http://dsw.sgupta-cloudera.com/api/altus-ds-1/models/call-model';
	$data = array('accessKey' => 'mcsm8oxnyfxzej7h1m2r4j1t68pv94u1', 'request' => array("feature" => "$age,$income,$usage,$sex,$state,$hh_members,$services"));
	$content = json_encode($data);
	$curl = curl_init($url);
	curl_setopt($curl, CURLOPT_HEADER, false);
	curl_setopt($curl, CURLOPT_RETURNTRANSFER, true);
	curl_setopt($curl, CURLOPT_HTTPHEADER,
	        array("Content-type: application/json"));
	curl_setopt($curl, CURLOPT_POST, true);
	curl_setopt($curl, CURLOPT_POSTFIELDS, $content);

	$json_response = curl_exec($curl);

	$status = curl_getinfo($curl, CURLINFO_HTTP_CODE);

	curl_close($curl);

	$response = json_decode($json_response, true);
        
	echo '<h3>Predicted Customer Spend:</h3> <h4>$',$response["response"]["result"],'</h4>';
?>
