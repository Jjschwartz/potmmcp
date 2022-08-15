#!/bin/bash
# This script downloads any new results from the server onto the local machine.
#
# It is specific to:

# SSH parameters
# The user@address for server, or name as specified in ~/.ssh/config
# if using password, you will need to modify this script to handle that
server=sococluster

# Result directories
local_dir=~/baposgmcp_results
remote_dir="~/baposgmcp_results"
env_dir_name=results

# List of environments to get results for
env_names=(Driving)

# Days in the past
period=-3

# Remote result exlude file
# = file containing files to exclude from download
remote_exclude_file="excluded_result_files2.txt"

echo "Getting latest results from server=$server"
echo "Only looking for results modified within the last $period days"
for env_name in ${env_names[*]}
do
	echo "Getting new results for environment=$env_name"

	remote_env_dir=$remote_dir/$env_name/$env_dir_name
	printf "Checking remote env dir=$remote_env_dir\n"
	# Use find command to look only for new files
	# refs:
	# https://unix.stackexchange.com/questions/10041/how-to-get-only-files-created-after-a-date-with-ls
	# https://serverfault.com/questions/354403/remove-path-from-find-command-output
    remote_files=($(ssh $server "find $remote_env_dir -mindepth 1 -maxdepth 1 -mtime $period -printf '%f\n' | sort -n"))
	printf "Remote env files=\n"
	printf "  %s\n" ${remote_files[*]}

	local_env_dir=$local_dir/$env_name/$env_dir_name
	printf "\nChecking local env dir=$local_env_dir\n"
	local_files=($(ls -1 $local_env_dir))
	printf "Local env files=\n"
	printf "  %s\n" ${local_files[*]}

	printf "\nGetting difference\n"
	# Using comm program which compares files
	# Hence the need to write to files
	printf "%s\n" ${local_files[*]} > /tmp/local_files
	printf "%s\n" ${remote_files[*]} > /tmp/remote_files
	file_diff=($(comm -13 /tmp/local_files /tmp/remote_files))
	printf "New remote files=\n"
	printf "  %s\n" ${file_diff[*]}

	printf "\nMaking local files for new remote files and copying files over\n"
	for file_name in ${file_diff[*]}
	do
		printf "Remote file %s\n" $file_name

		read -p "Would you like to download this file [y/n]? " -n 1 -r
		echo    # (optional) move to a new line
		if [[ $REPLY =~ ^[Yy]$ ]]
		then
			printf "  Compressing desired remote results files\n"
			ssh $server "tar -cz --exclude-from=baposgmcp_results/$remote_exclude_file -C $remote_env_dir -f $file_name.tar  $file_name"
			printf "  Copying tarball to local\n"
			scp $server:$file_name.tar $local_env_dir/
			printf "  Extracting tarball\n"
			tar -xf $local_env_dir/$file_name.tar -C $local_env_dir
			printf "  Cleaning up\n"
			ssh $server "rm $file_name.tar"
			rm $local_env_dir/$file_name.tar
		fi
	done
done

printf "Et Voila!"
