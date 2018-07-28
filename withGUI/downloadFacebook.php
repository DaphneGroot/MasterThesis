<?php

require_once __DIR__ . '/vendor/autoload.php'; // change path as needed

$fb = new Facebook\Facebook([
	'app_id' => '147466205907909',
	'app_secret' => 'c3c65e4b171f369cc05cb39551891e30',
	'default_graph_version' => 'v2.11',
]);

set_error_handler (
    function($errno, $errstr, $errfile, $errline) {
        throw new ErrorException($errstr, $errno, 0, $errfile, $errline);     
    }
);



$accessToken = "147466205907909|Zwsg3A2Q4pC6dOBVMzOZ5Nr7U40";

$postID = $argv[1];
$source = $argv[2];

//GET NEWSITEMS
try {
	$response = $fb->get('/'.$source, $accessToken);
} catch(Facebook\Exceptions\FacebookResponseException $e) {
	echo 'Graph returned an error: ' . $e->getMessage();
	exit;
} catch(Facebook\Exceptions\FacebookSDKException $e) {
	echo 'Facebook SDK returned an error: ' . $e->getMessage();
	exit;
}

$sourceInformation = $response->getGraphNode();
$sourceID = $sourceInformation->getField('id');

// echo "$sourceID\n";

try {
	$response = $fb->get('/'.$sourceID.'_'.$postID.'?fields=message,link', $accessToken);
} catch(Facebook\Exceptions\FacebookResponseException $e) {
	echo 'Graph returned an error: ' . $e->getMessage();
	exit;
} catch(Facebook\Exceptions\FacebookSDKException $e) {
	echo 'Facebook SDK returned an error: ' . $e->getMessage();
	exit;
}

$postInformation = $response->getGraphNode();
$message = $postInformation->getField('message');
$link = $postInformation->getField('link');



echo "$message\n$link";



?>