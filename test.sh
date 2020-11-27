num_test=$1
perc_cases=0
for n in {1..9}
do
	perc_cases=$(expr $perc_cases + 10)
	tot_active=0
	tot_random=0
	for i in $(seq 1 $num_test)
	do
		res=$(python3 main.py $perc_cases $2)
		#tot=$(python3 -c "print($tot+$res)")
		res_active=$(echo $res | cut -d ';' -f 1)
		res_random=$(echo $res | cut -d ';' -f 2)
		tot_active=$(python3 -c "print($tot_active + $res_active)")
		tot_random=$(python3 -c "print($tot_random + $res_random)")
	done
	mean_active=$(python3 -c "print($tot_active/$num_test)")
	mean_random=$(python3 -c "print($tot_random/$num_test)")
	printf "Mean of rate of accuracy in the active learner's tree = $mean_active\nMean of rate of accuracy in the random learner's tree = $mean_random\nRate of paths perfomed on the total number of leaves in the tree=$perc_cases \n\n"
done