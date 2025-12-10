# Restoring can be dangerous, so do not allow it by default
default[:imos_backup][:restore][:allow] = false

# Where to take backups from when restoring
default[:imos_backup][:restore][:from_host] = "s3://imos-backups"

default[:imos_backup][:restore][:directives] = []

# User to use for restoration coming from remote server
default[:imos_backup][:restore][:username] = "restore"
