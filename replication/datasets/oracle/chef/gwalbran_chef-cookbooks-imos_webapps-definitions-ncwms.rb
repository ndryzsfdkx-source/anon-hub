#
# Cookbook Name:: imos_webapps
# Definition:: ncwms
#
# Copyright 2012, IMOS
#
# All rights reserved - Do Not Redistribute
#

define :ncwms do

  app_parameters          = params[:app_parameters]
  instance_parameters     = params[:instance_parameters]
  instance_service_name   = params[:instance_service_name]
  instance_base_directory = params[:instance_base_directory]

  app_name                = app_parameters['name']

  # Overriding the WMS-servlet.xml file extracted by the WAR
  wms_servlet_xml = "#{instance_base_directory}/webapps/#{app_name}/WEB-INF/WMS-servlet.xml"

  template wms_servlet_xml do
    source   "#{app_name}/WMS-servlet.xml.erb"
    owner    node['tomcat']['user']
    group    node['tomcat']['group']
    mode     0644
    backup   3
    notifies :restart, "service[#{instance_service_name}]", :delayed

    variables({
      :cache_dir => app_parameters['cache_dir']
    })
  end

  # On dev machines, we'll create the ncwms cache dir, so it doesn't fail
  # On real production machines we'd like it to fail if the directory is not in
  # place
  if Chef::Config[:dev]
    directory app_parameters['cache_dir'] do
      owner     node['tomcat']['user']
      group     node['tomcat']['group']
      mode      0755
      recursive true
    end
  end

  password = Chef::EncryptedDataBagItem.load('passwords', 'ncwms')['password']

  # NcWMS is making funny things with config, such as updating the config file
  # while running. We want to be able to manage the config from chef so the way
  # we do it is that if the chef config have changed - we'll trigger a restart.
  # However if the chef config is different from what's on the node and the chef
  # config was unchanged - we'll avoid restarting NcWMS.
  ncwms_config_real = ::File.join(app_parameters['cache_dir'], "config.xml")
  ncwms_config_from_chef = ::File.join(app_parameters['cache_dir'], "config_base.xml")

  template ncwms_config_from_chef do
    cookbook "external_templates"
    source   "#{app_name}/config.xml.erb"
    owner    node['tomcat']['user']
    group    node['tomcat']['group']
    mode     00644
    notifies :create, "ruby_block[refresh_ncwms_config_#{instance_service_name}]", :immediately
    variables ({
      :root     => app_parameters['root'],
      :password => password
    })
  end

  ruby_block "refresh_ncwms_config_#{instance_service_name}" do
    block do
      FileUtils.cp ncwms_config_from_chef, ncwms_config_real
      FileUtils.chown node['tomcat']['user'], node['tomcat']['group'], ncwms_config_real
    end
    action   :nothing
    notifies :restart, "service[#{instance_service_name}]", :delayed
  end

end

